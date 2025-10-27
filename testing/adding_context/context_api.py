#!/usr/bin/env python3
"""
context_api.py — public library API + CLI for querying **base-text** windows
from a cached corpus, using the enhanced behavior:
  - windows centered on the term’s base span,
  - ranked by closeness to target length (2*window + len(term_in_base_tokens)),
  - minimal prettify (HTML unescape, zero-width removal, markdown de-escape).

Usage (CLI):
  # Single term
  python context_api.py --cache cache/ngram_df_baseTokens_and_ngramText.parquet \
                        --term botox_procedure \
                        -k 5 \
                        --window 5 \
                        --format parquet \
                        --json

  # Multi-term (comma-separated)
  python context_api.py --cache cache/ngram_df_baseTokens_and_ngramText.parquet \
                        --terms botox_procedure,throatox \
                        -k 3 \
                        --window 5 \
                        --format parquet \
                        --json
"""

from typing import List, Dict, Sequence, Union
import argparse
import json
import pandas as pd
import sys

# If your repo layout requires this path tweak, keep it;
# otherwise it's harmless.
sys.path.append('../vocal_disorder')

# Fixed module import (no dynamic import or --module arg needed)
from testing.adding_context import test_get_similar as setup_mod

__all__ = [
    "load_cached_corpus",
    "query_base_windows",
    "query_term_windows",
    "query_from_cache",
    "query_base_windows_for_terms",
    "query_from_cache_for_terms",
    "windows_list_for_terms",
    "windows_list_from_cache_for_terms",
    "select_whole_docs_by_length",
    "select_whole_docs_by_length_from_cache",
]


# ---------------- Core helpers ----------------

def load_cached_corpus(
    cache_path: str,
    *,
    format: str = "parquet",
) -> pd.DataFrame:
    """Load cached dataframe (doc_id, base_tokens, ngram_text) via setup_mod."""
    return setup_mod.load_ngram_df(cache_path, format=format)


def query_base_windows(
    df: pd.DataFrame,
    term: str,
    k: int,
    window: int,
) -> List[List[str]]:
    """
    Return **base-text** windows grouped per document (up to k docs), using the improved logic:
      - windows centered on the term’s mapped base span,
      - documents ranked by closest achievable window length to (2*window + term_len_in_base_tokens),
      - per-doc windows deduplicated and ordered by closeness.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if window < 0:
        raise ValueError("window must be >= 0")
    return setup_mod.sample_base_windows_around_term(df, term, k, window)


# Back-compat alias (old name)
def query_term_windows(
    df: pd.DataFrame,
    term: str,
    k: int,
    window: int,
) -> List[List[str]]:
    """Deprecated: use query_base_windows. Kept for compatibility."""
    return query_base_windows(df, term, k, window)


def query_from_cache(
    cache_path: str,
    term: str,
    k: int,
    window: int,
    *,
    format: str = "parquet",
) -> List[List[str]]:
    """Convenience: load cache and query base windows in one call."""
    df = load_cached_corpus(cache_path, format=format)
    return query_base_windows(df, term, k, window)


# ---------- Multi-term helpers ----------

def query_base_windows_for_terms(
    df: pd.DataFrame,
    terms: Sequence[str],
    k: int,
    window: int,
) -> Dict[str, List[List[str]]]:
    """
    Query base windows for multiple terms at once.
    Returns a dict mapping each term -> List[List[str]] (same per-doc windows structure).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if window < 0:
        raise ValueError("window must be >= 0")

    out: Dict[str, List[List[str]]] = {}
    for term in terms:
        out[term] = setup_mod.sample_base_windows_around_term(df, term, k, window)
    return out


def query_from_cache_for_terms(
    cache_path: str,
    terms: Sequence[str],
    k: int,
    window: int,
    *,
    format: str = "parquet",
) -> Dict[str, List[List[str]]]:
    """
    Convenience: load cache once, then query windows for multiple terms.
    Returns {term: List[List[str]]}.
    """
    df = load_cached_corpus(cache_path, format=format)
    return query_base_windows_for_terms(df, terms, k, window)


# ---------- Convenience: list-style outputs ----------

def windows_list_for_terms(
    df: pd.DataFrame,
    terms: Sequence[str],
    k: int,
    window: int,
    *,
    flatten: bool = False,
) -> Union[List[List[str]], List[List[List[str]]]]:
    """
    Convenience wrapper that returns windows as a List rather than a dict.

    If flatten=False (default):
        returns a List[List[List[str]]] where each element aligns with `terms` order:
            [ windows_for_term_0, windows_for_term_1, ... ]
        and each windows_for_term_i is the usual per-doc List[List[str]].

    If flatten=True:
        flattens across terms into a single List[List[str]] (concatenating per-doc windows).
    """
    result_per_term = [query_base_windows(df, t, k, window) for t in terms]
    if not flatten:
        return result_per_term
    flat: List[List[str]] = []
    for per_doc in result_per_term:
        flat.extend(per_doc)
    return flat


def windows_list_from_cache_for_terms(
    cache_path: str,
    terms: Sequence[str],
    k: int,
    window: int,
    *,
    format: str = "parquet",
    flatten: bool = False,
) -> Union[List[List[str]], List[List[List[str]]]]:
    """
    Same as windows_list_for_terms, but loads the cache internally first.
    """
    df = load_cached_corpus(cache_path, format=format)
    return windows_list_for_terms(df, terms, k, window, flatten=flatten)

# --- paste this anywhere below the multi-term helpers (before CLI is fine) ---

from typing import List  # already imported at top, keep if present

def select_whole_docs_by_length(
    df: pd.DataFrame,
    term: str,
    k: int,
    target_len: int,
) -> List[str]:
    """
    Return up to `k` full documents (as a single space-joined string of base tokens)
    for which the processed n-gram token `term` appears, ranked by how close the
    document length (number of base tokens) is to `target_len`.

    Selection:
      1) Filter to rows whose n-gram tokens contain `term`.
      2) Compute base_len = len(base_tokens).
      3) Rank by |base_len - target_len| ascending; tie-break by larger base_len.
      4) Return [" ".join(base_tokens), ...] up to k.
    """
    if k <= 0:
        return []
    if target_len < 1:
        raise ValueError("target_len must be >= 1")

    work = df.copy(deep=False)

    # Ensure ngram_token_set exists for fast membership checks
    if "ngram_token_set" not in work.columns:
        if "ngram_tokens" in work.columns:
            work["ngram_token_set"] = work["ngram_tokens"].apply(set)
        elif "ngram_text" in work.columns:
            work["ngram_tokens"] = work["ngram_text"].astype(str).str.split()
            work["ngram_token_set"] = work["ngram_tokens"].apply(set)
        else:
            raise ValueError("Need 'ngram_tokens' or 'ngram_text' in DataFrame.")

    if "base_tokens" not in work.columns:
        raise ValueError("Need 'base_tokens' in DataFrame.")

    # Filter by presence of term in n-gram tokens
    hits = work[work["ngram_token_set"].apply(lambda s: term in s)].copy()
    if hits.empty:
        return []

    # Compute base_len if missing
    if "base_len" not in hits.columns:
        hits["base_len"] = hits["base_tokens"].apply(len)

    # Score and rank
    hits["score"] = (hits["base_len"] - int(target_len)).abs()
    hits = hits.sort_values(["score", "base_len"], ascending=[True, False]).head(k)

    # Return full base docs as strings
    return [" ".join(toks) for toks in hits["base_tokens"].tolist()]


def select_whole_docs_by_length_from_cache(
    cache_path: str,
    term: str,
    k: int,
    target_len: int,
    *,
    format: str = "parquet",
) -> List[str]:
    """
    Convenience wrapper: load cache, then call select_whole_docs_by_length.
    """
    df = load_cached_corpus(cache_path, format=format)
    return select_whole_docs_by_length(df, term, k, target_len)


# ---------------- CLI ----------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Query **base-text** windows around term(s) from a cached n-gram dataframe."
    )
    p.add_argument("--cache", default="testing/adding_context/cache/ngram_df_baseTokens_and_ngramText.parquet",
                   help="Path to cached DF (parquet/feather/pickle).")
    # Either --term or --terms (comma-separated)
    p.add_argument("--term",
                   help="Exact n-gram token to search (e.g., botox_procedure).")
    p.add_argument("--terms",
                   help="Comma-separated list of terms (e.g., termA,termB,termC).")
    p.add_argument("-k", type=int, required=True,
                   help="Max number of matching documents to return per term.")
    p.add_argument("--window", type=int, required=True,
                   help="Context window size on each side (in base tokens).")
    p.add_argument("--format", default="parquet",
                   choices=["parquet", "feather", "pickle"],
                   help="Cache format.")
    p.add_argument("--json", action="store_true",
                   help="Print JSON to stdout (default is a readable text format).")
    return p


def _print_human_single(out: List[List[str]]) -> None:
    if not out:
        print("[]")
        return
    for i, winlist in enumerate(out, 1):
        print(f"Doc {i} — {len(winlist)} hits")
        for w in winlist:
            print(f"  - {w}")


def _print_human_multi(out: Dict[str, List[List[str]]]) -> None:
    if not out:
        print("{}")
        return
    for term, win_per_doc in out.items():
        print(f"[{term}]")
        if not win_per_doc:
            print("  []")
            continue
        for i, winlist in enumerate(win_per_doc, 1):
            print(f"  Doc {i} — {len(winlist)} hits")
            for w in winlist:
                print(f"    - {w}")


def main():
    args = _build_argparser().parse_args()

    # Validate term arguments
    terms: Sequence[str]
    if args.terms:
        terms = [t.strip() for t in args.terms.split(",") if t.strip()]
        if not terms:
            raise SystemExit("Error: --terms provided but no valid terms found after parsing.")
    elif args.term:
        terms = [args.term]
    else:
        raise SystemExit("Error: provide either --term or --terms.")

    df = load_cached_corpus(args.cache, format=args.format)

    if len(terms) == 1:
        out_single = query_base_windows(df, terms[0], args.k, args.window)
        if args.json:
            print(json.dumps(out_single, ensure_ascii=False))
        else:
            _print_human_single(out_single)
    else:
        out_multi = query_base_windows_for_terms(df, terms, args.k, args.window)
        if args.json:
            print(json.dumps(out_multi, ensure_ascii=False))
        else:
            _print_human_multi(out_multi)


if __name__ == "__main__":
    
    df = load_cached_corpus("base_ngram_cache_with_details.parquet", format="parquet")
    docs = select_whole_docs_by_length(df, term="choke_sensation", k=2, target_len=88)
    for i, d in enumerate(docs, 1):
        print(d)

    