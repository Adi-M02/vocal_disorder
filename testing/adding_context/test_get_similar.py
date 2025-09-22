#!/usr/bin/env python3
from typing import List, Tuple
import pandas as pd
import re
import sys
import os
import html
from difflib import SequenceMatcher

# --- pipeline pieces 
from utils.load_process_ngram_docs import process_ngram_docs
from utils.load_and_process_docs import process_all_noburp
from utils.load_json import load_json

# Provided utilities (paths per your message)
from tokenizer import clean_and_tokenize               # testing/adding_context/tokenizer.py
from utils.load_lemmatizer import load_lookup          # ../vocal_disorder/utils/load_lemmatizer.py (placeholder)
from spellchecker_folder.spellchecker import spellcheck_token_list
from utils.stopwords import STOPWORDS                  # Not used when stoplist=False; imported for parity

# If you use lemmas, point to your JSON (placeholder path OK)
LOOKUP_PATH = "combined_lemmas.json"
_LOOKUP = None
if os.path.exists(LOOKUP_PATH):
    try:
        _LOOKUP = load_lookup(LOOKUP_PATH)
    except Exception:
        _LOOKUP = None


# =============================================================================
# BUILD
# =============================================================================

def build_ngram_df(ngram_phraser_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - doc_id          : int
      - base_tokens     : list[str]   (split tokens; per your request)
      - base_len        : int         (# of base tokens; for sorting)
      - ngram_tokens    : list[str]   (processed tokens with n-grams)
      - ngram_text      : str         (space-joined n-gram tokens; persisted)
      - ngram_token_set : set[str]    (for fast membership on n-gram tokens)
      - ngram_len       : int         (# of n-gram tokens; for sorting)
    """
    # 1) Base tokens (split tokens). If the function returns strings, split them.
    base_docs = process_all_noburp(tokenize=False, stoplist=False, lemmatize=False)
    if base_docs and isinstance(base_docs[0], str):
        base_docs = [s.split() for s in base_docs]

    # 2) N-gram token lists (existing pipeline)
    ngram_docs = process_ngram_docs(ngram_phraser_dir)

    if len(base_docs) != len(ngram_docs):
        raise ValueError(f"Mismatched lengths: base={len(base_docs)} vs ngram={len(ngram_docs)}")

    df = pd.DataFrame({
        "doc_id": range(len(base_docs)),
        "base_tokens": base_docs,
        "ngram_tokens": ngram_docs,
    })

    # Derived columns
    df["base_len"] = df["base_tokens"].apply(len)
    df["ngram_text"] = df["ngram_tokens"].apply(" ".join)
    df["ngram_token_set"] = df["ngram_tokens"].apply(set)
    df["ngram_len"] = df["ngram_tokens"].apply(len)
    return df


# =============================================================================
# SAVE / LOAD (persist base_tokens + ngram_text only)
# =============================================================================

def save_ngram_df(df: pd.DataFrame, path: str, *, format: str = "parquet") -> None:
    """
    Persist a compact cache containing ONLY:
      - doc_id
      - base_tokens  (list[str], split tokens)
      - ngram_text   (space-joined processed tokens)

    format: 'parquet' (default) | 'feather' | 'pickle'
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_save = df[["doc_id", "base_tokens", "ngram_text"]].copy()

    if format == "parquet":
        to_save.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    elif format == "feather":
        to_save.reset_index(drop=True).to_feather(path)
    elif format == "pickle":
        to_save.to_pickle(path, protocol=5)
    else:
        raise ValueError("format must be one of {'parquet','feather','pickle'}")


def load_ngram_df(path: str, *, format: str = "parquet") -> pd.DataFrame:
    """
    Load the cached dataframe and re-materialize:
      - base_len
      - ngram_tokens, ngram_token_set, ngram_len
    """
    if format == "parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    elif format == "feather":
        df = pd.read_feather(path)
    elif format == "pickle":
        df = pd.read_pickle(path)
    else:
        raise ValueError("format must be one of {'parquet','feather','pickle'}")

    # Rebuild derived cols
    df["base_len"] = df["base_tokens"].apply(len)
    df["ngram_tokens"] = df["ngram_text"].astype(str).str.split()
    df["ngram_token_set"] = df["ngram_tokens"].apply(set)
    df["ngram_len"] = df["ngram_tokens"].apply(len)
    return df


# =============================================================================
# BASIC COUNTS + NGRAM WINDOWS
# =============================================================================

def count_docs_containing(df: pd.DataFrame, term: str) -> int:
    """Number of docs whose N-GRAM token set contains the exact term."""
    return int(df["ngram_token_set"].apply(lambda s: term in s).sum())


def sample_docs_containing(df: pd.DataFrame, term: str, k: int, window: int) -> List[List[str]]:
    """
    Return windows of context around `term` from up to `k` matching documents (N-GRAM tokens).

    For each matching document, every occurrence of `term` produces:
        <window tokens before> + [term] + <window tokens after>

    Returns: list (one element per document), where each element is a list of window strings.
    """
    if window < 0:
        raise ValueError("`window` must be >= 0")
    if "ngram_token_set" not in df.columns or "ngram_tokens" not in df.columns:
        raise ValueError("DataFrame must include 'ngram_tokens' and 'ngram_token_set' columns.")

    mask = df["ngram_token_set"].apply(lambda s: term in s)
    hits = df[mask].copy()
    if hits.empty:
        return []

    if "ngram_len" not in hits.columns:
        hits["ngram_len"] = hits["ngram_tokens"].apply(len)

    hits = hits.sort_values("ngram_len", ascending=False).head(k)

    results: List[List[str]] = []
    for _, row in hits.iterrows():
        toks: List[str] = row["ngram_tokens"]
        n = len(toks)
        occ_windows: List[str] = []
        for i, tok in enumerate(toks):
            if tok == term:
                left = max(0, i - window)
                right = min(n, i + window + 1)  # include term itself
                occ_windows.append(" ".join(toks[left:right]))
        if occ_windows:
            results.append(occ_windows)
    return results


# =============================================================================
# PIPELINE-ONLY BASE WINDOWS (ranked by closest length) + PRETTIFY
# =============================================================================

# light normalizer for aligning processed tokens back to base (handles case/punct/digits/plurals)
_punct_re = re.compile(r"[^\w\s]+", re.UNICODE)
_md_escape_re = re.compile(r'\\([\\`*_{}\[\]()<>#+\-.!])')   # remove markdown backslash-escapes
_standalone_emph_re = re.compile(r'(?<!\w)[*_]{1,3}(?!\w)')  # drop stray *, **, *** and _ outside words

def _norm(tok: str) -> str:
    s = tok.lower()
    s = _punct_re.sub("", s)         # drop punctuation
    s = re.sub(r"\d+", "", s)        # drop digits
    # Conservative plural folding helps 'procedure' ~ 'procedures'
    if len(s) > 4 and s.endswith("es"):
        s = s[:-2]
    elif len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s

def _prettify_base_snippet(text: str) -> str:
    """
    Minimal cleanup so output looks closer to rendered Reddit text:
      - HTML-unescape (e.g., &#x200B; → \u200B)
      - remove zero-width / BOM
      - drop markdown backslash-escapes (\"\\*\" → \"*\")
      - remove standalone emphasis markers (*, **, ***) that aren't part of words
      - collapse whitespace
    """
    s = html.unescape(text)
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = _md_escape_re.sub(r"\1", s)
    s = _standalone_emph_re.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pipeline_tokens_from_base_tokens(base_tokens: List[str]) -> List[str]:
    """
    Recreate your pre-ngram pipeline with stoplist=False using the provided utils.
    """
    text = " ".join(base_tokens)

    toks = clean_and_tokenize(text)
    if _LOOKUP is not None:
        toks = [_LOOKUP.get(tok, tok) for tok in toks]

    text = " ".join(toks)
    toks = clean_and_tokenize(text)

    toks = spellcheck_token_list(toks)

    if _LOOKUP is not None:
        toks = [_LOOKUP.get(tok, tok) for tok in toks]

    text = " ".join(toks)
    toks = clean_and_tokenize(text)
    return toks


def _align_proc_to_base(base_tokens: List[str], proc_tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Align each pre-ngram 'proc' token back to a single base token index span.
    Guarantees non-empty spans when base is non-empty (prevents "" windows).
    """
    spans: List[Tuple[int, int]] = []
    base_norm = [_norm(b) for b in base_tokens]
    i = 0
    n_base = len(base_tokens)

    def _one_token_span(ix: int) -> Tuple[int, int]:
        if n_base == 0:
            return (0, 0)
        ix = min(max(ix, 0), n_base - 1)
        return (ix, ix + 1)

    for ptok in proc_tokens:
        target = _norm(ptok)
        if not target:
            spans.append(_one_token_span(i))
            i = min(i + 1, n_base)
            continue

        # exact forward scan
        matched = False
        j = i
        while j < n_base:
            if base_norm[j] == target:
                spans.append((j, j + 1))
                i = j + 1
                matched = True
                break
            j += 1
        if matched:
            continue

        # fuzzy scan (limited window)
        best_span = None
        best_score = 0.0
        scan_start = i
        scan_end = min(n_base, i + 50)
        for s in range(scan_start, scan_end):
            cand = base_norm[s]
            score = SequenceMatcher(None, cand, target).ratio()
            if score > best_score:
                best_score, best_span = score, (s, s + 1)
        if best_span and best_score >= 0.80:
            spans.append(best_span)
            i = best_span[1]
        else:
            spans.append(_one_token_span(i))
            i = min(i + 1, n_base)

    return spans


def _ngram_to_proc_spans(proc_tokens: List[str], ngram_tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Map each n-gram token to a span over pre-ngram 'proc_tokens'.
    Uses exact part matching with a greedy forward scan (parts are ng token split by '_').
    Fallback: if parts not found contiguously, map to a 1-token span at current pointer.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(proc_tokens)

    for ng in ngram_tokens:
        parts = ng.split("_")
        m = len(parts)
        if m == 0:
            if i < n:
                spans.append((i, i + 1)); i += 1
            else:
                spans.append((n, n))
            continue

        matched = False
        start_i = i
        while start_i + m <= n:
            if proc_tokens[start_i:start_i + m] == parts:
                spans.append((start_i, start_i + m))
                i = start_i + m
                matched = True
                break
            start_i += 1

        if not matched:
            if i < n:
                spans.append((i, i + 1)); i += 1
            else:
                spans.append((n, n))

    return spans


def _compose_spans(proc2base: List[Tuple[int, int]], ng2proc: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Compose ngram->proc spans with proc->base spans to produce ngram->base spans.
    Guarantees non-empty base spans when possible.
    """
    out: List[Tuple[int, int]] = []
    m = len(proc2base)
    for (pL, pR) in ng2proc:
        if pL >= pR or pL >= m:
            # fallback: snap to last token if available
            if m > 0:
                out.append((max(0, proc2base[-1][0]), proc2base[-1][1]))
            else:
                out.append((0, 0))
            continue
        # union of constituent proc token spans
        bL = proc2base[pL][0]
        bR = proc2base[min(pR - 1, m - 1)][1]
        if bR <= bL and m > 0:
            # ensure at least 1 token
            bR = min(bL + 1, proc2base[-1][1])
        out.append((bL, bR))
    return out


def sample_base_windows_around_term(df: pd.DataFrame, term: str, k: int, window: int) -> List[List[str]]:
    """
    PIPELINE-ONLY: Return base-text windows around an n-gram `term`, selecting up to `k` **documents**
    whose windows are **closest in length** to the target (2*window + term_base_len).

    Windows are prettified (html-unescape, zero-width remove, markdown de-escape, whitespace collapse).
    """
    if window < 0:
        raise ValueError("`window` must be >= 0")

    work = df.copy()

    if "ngram_tokens" not in work.columns:
        if "ngram_text" not in work.columns:
            raise ValueError("Need 'ngram_tokens' or 'ngram_text'.")
        work["ngram_tokens"] = work["ngram_text"].astype(str).str.split()

    if "base_tokens" not in work.columns:
        raise ValueError("Need 'base_tokens' in DataFrame (you persist split tokens).")

    if "base_len" not in work.columns:
        work["base_len"] = work["base_tokens"].apply(len)

    # Filter to docs that actually contain the exact n-gram token
    mask = work["ngram_tokens"].apply(lambda toks: term in toks)
    hits = work[mask].copy()
    if hits.empty:
        return []

    # Pre-compute the target window length in base tokens (parts count is a decent proxy for term length)
    term_parts_len = max(1, len([p for p in term.split("_") if p]))
    target_len = 2 * window + term_parts_len

    ranked_docs = []  # list of (score, -base_len, windows_list)
    for _, row in hits.iterrows():
        base_tokens: List[str] = row["base_tokens"]
        ngram_tokens: List[str] = row["ngram_tokens"]

        n_base = len(base_tokens)
        doc_candidates: List[Tuple[str, int]] = []  # (snippet, length)

        # ---- (A) Direct base match if possible ----
        base_norm = [_norm(t) for t in base_tokens]
        parts = [_norm(p) for p in term.split("_") if _norm(p)]
        m = len(parts)

        spans: List[Tuple[int, int]] = []
        if m > 0 and n_base > 0:
            i = 0
            while i + m <= len(base_norm):
                if base_norm[i:i+m] == parts:
                    spans.append((i, i + m))
                    i += m
                else:
                    i += 1

        if spans:
            for (bL, bR) in spans:
                left  = max(0, bL - window)
                right = min(n_base, bR + window)
                snippet_tokens = base_tokens[left:right]
                snippet = _prettify_base_snippet(" ".join(snippet_tokens))
                if snippet:
                    doc_candidates.append((snippet, len(snippet_tokens)))
        else:
            # ---- (B) Fallback: pipeline compose to locate term span ----
            proc_tokens = _pipeline_tokens_from_base_tokens(base_tokens)
            proc2base   = _align_proc_to_base(base_tokens, proc_tokens)
            ng2proc     = _ngram_to_proc_spans(proc_tokens, ngram_tokens)
            ng2base     = _compose_spans(proc2base, ng2proc)

            positions = [j for j, tok in enumerate(ngram_tokens) if tok == term]
            for pos in positions:
                bL_term, bR_term = ng2base[pos]
                if n_base == 0:
                    continue
                bL_term = min(max(bL_term, 0), n_base - 1)
                bR_term = max(min(bR_term, n_base), bL_term + 1)
                left  = max(0, bL_term - window)
                right = min(n_base, bR_term + window)
                snippet_tokens = base_tokens[left:right]
                snippet = _prettify_base_snippet(" ".join(snippet_tokens))
                if snippet:
                    doc_candidates.append((snippet, len(snippet_tokens)))

        # Deduplicate candidates by text, keep first occurrence (and its length)
        if doc_candidates:
            seen = set()
            dedup: List[Tuple[str, int]] = []
            for s, L in doc_candidates:
                if s not in seen:
                    dedup.append((s, L)); seen.add(s)

            # Sort windows in this doc by closeness to target, ascending
            dedup.sort(key=lambda x: abs(x[1] - target_len))

            # Doc score = best (closest) window distance
            doc_score = abs(dedup[0][1] - target_len)
            ranked_docs.append((doc_score, -n_base, [s for s, _ in dedup]))

    if not ranked_docs:
        return []

    # Pick up to k docs with smallest score (tie-break by longer base_len)
    ranked_docs.sort(key=lambda x: (x[0], x[1]))
    top = ranked_docs[:k]

    # Return only the ordered windows per document (same shape as before)
    return [winlist for _, __, winlist in top]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Build fresh (adjust path as needed)
    df = build_ngram_df("testing/ngram_evals_test_no_digits/4")

    # Save compact cache (doc_id + base_tokens + ngram_text)
    cache_path = "cache/ngram_df_baseTokens_and_ngramText.parquet"
    save_ngram_df(df, cache_path, format="parquet")

    # Load and sample
    df2 = load_ngram_df(cache_path, format="parquet")

    TERM = "throatox"
    print("N-gram windows:", sample_docs_containing(df2, TERM, 3, window=50))
    print("Base windows (ranked by closeness):", sample_base_windows_around_term(df2, TERM, 3, window=50))

