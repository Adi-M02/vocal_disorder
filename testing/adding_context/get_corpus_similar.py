from typing import Optional, List
import pandas as pd
import re
import sys
import os

sys.path.append("../vocal_disorder")  # adjust as needed
from utils.load_process_ngram_docs import process_ngram_docs
from utils.load_and_process_docs import process_all_noburp
from utils.load_json import load_json

def build_ngram_df(ngram_phraser_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - doc_id          : int
      - base_tokens     : list[str]   (tokenized 'base' form; no lemmatization)   [NOT persisted]
      - base_text       : str         (space-joined base tokens)                  [NOT persisted]
      - base_len        : int         (# of base tokens; used for sorting)        [NOT persisted]
      - ngram_tokens    : list[str]   (processed tokens with n-grams)
      - ngram_text      : str         (space-joined n-gram tokens)
      - ngram_token_set : set[str]    (for fast membership on n-gram tokens)
    """
    base_docs = process_all_noburp(stoplist=False, lemmatize=False)
    ngram_docs = process_ngram_docs(ngram_phraser_dir)

    if len(base_docs) != len(ngram_docs):
        raise ValueError(f"Mismatched lengths: base={len(base_docs)} vs ngram={len(ngram_docs)}")

    df = pd.DataFrame({
        "doc_id": range(len(base_docs)),
        "base_tokens": base_docs,
        "ngram_tokens": ngram_docs,
    })
    df["base_text"] = df["base_tokens"].apply(" ".join)
    df["ngram_text"] = df["ngram_tokens"].apply(" ".join)
    df["ngram_token_set"] = df["ngram_tokens"].apply(set)
    df["base_len"] = df["base_tokens"].apply(len)
    return df

# ---- SAVE (only ngram_text) ----
def save_ngram_df(df: pd.DataFrame, path: str, *, format: str = "parquet") -> None:
    """
    Persist a lightweight cache containing ONLY:
      - doc_id
      - ngram_text  (space-joined processed tokens)

    format: 'parquet' (default) | 'feather' | 'pickle'
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_save = df[["doc_id", "ngram_text"]].copy()

    if format == "parquet":
        to_save.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    elif format == "feather":
        to_save.reset_index(drop=True).to_feather(path)
    elif format == "pickle":
        to_save.to_pickle(path, protocol=5)
    else:
        raise ValueError("format must be one of {'parquet','feather','pickle'}")

# ---- LOAD (rebuild everything from ngram_text) ----
def load_ngram_df(path: str, *, format: str = "parquet") -> pd.DataFrame:
    """
    Load the cached dataframe (doc_id, ngram_text) and re-materialize:
      - ngram_tokens, ngram_token_set
      - ngram_len (len of ngram_tokens)
    If base_* columns are absent (they will be), sample_docs_containing will fall back to ngram_*.
    """
    if format == "parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    elif format == "feather":
        df = pd.read_feather(path)
    elif format == "pickle":
        df = pd.read_pickle(path)
    else:
        raise ValueError("format must be one of {'parquet','feather','pickle'}")

    # Rebuild token-level structures from ngram_text
    # Assumes ngram_text is space-joined tokens produced by your pipeline.
    df["ngram_tokens"] = df["ngram_text"].str.split()
    df["ngram_token_set"] = df["ngram_tokens"].apply(set)
    df["ngram_len"] = df["ngram_tokens"].apply(len)
    return df

def count_docs_containing(df: pd.DataFrame, term: str) -> int:
    """Number of docs whose N-GRAM token set contains the exact term."""
    return int(df["ngram_token_set"].apply(lambda s: term in s).sum())

def sample_docs_containing(df: pd.DataFrame, term: str, k: int, window: int) -> List[List[str]]:
    """
    Return windows of context around `term` from up to `k` matching documents.

    For each matching document, every occurrence of `term` (matched as an exact n-gram token)
    produces a separate window consisting of:
        <window tokens before> + [term] + <window tokens after>

    The function returns a list with one element per *document* (up to k docs, sorted by length desc);
    each element is a list of window strings for that document, e.g.:
        [
          ["... w5 w4 w3 w2 w1 term w1 w2 w3 w4 w5 ...",  "... another occurrence ..."],
          ["..."],
          ...
        ]

    Notes:
    - Uses `ngram_tokens` and `ngram_token_set`. If `ngram_len` is not present, it is computed on the fly.
    - `window` must be >= 0 and is required (no full-document fallback).
    """
    if window < 0:
        raise ValueError("`window` must be >= 0")

    if "ngram_token_set" not in df.columns or "ngram_tokens" not in df.columns:
        raise ValueError("DataFrame must include 'ngram_tokens' and 'ngram_token_set' columns.")

    # filter to docs that contain the term as an exact n-gram token
    mask = df["ngram_token_set"].apply(lambda s: term in s)
    hits = df[mask].copy()
    if hits.empty:
        return []

    # ensure we can sort by length (prefer existing ngram_len, else compute)
    if "ngram_len" not in hits.columns:
        hits["ngram_len"] = hits["ngram_tokens"].apply(len)

    # sort by length desc, then take up to k docs
    hits = hits.sort_values("ngram_len", ascending=False).head(k)

    results: List[List[str]] = []
    for _, row in hits.iterrows():
        tokens: List[str] = row["ngram_tokens"]
        n = len(tokens)
        # collect windows for every occurrence of `term`
        occ_windows: List[str] = []
        for i, tok in enumerate(tokens):
            if tok == term:
                left = max(0, i - window)
                right = min(n, i + window + 1)  # +1 to include the term itself boundary
                occ_tokens = tokens[left:right]
                occ_windows.append(" ".join(occ_tokens))
        # Only append docs that truly yielded at least one window (paranoid check)
        if occ_windows:
            results.append(occ_windows)

    return results

if __name__ == "__main__":
    # Build fresh
    df = build_ngram_df("testing/ngram_evals_test_no_digits/4")
<<<<<<< Updated upstream
    # Save ONLY ngram_text
    cache_path = "cache/ngram_df_only_text.parquet"
    save_ngram_df(df, cache_path, format="parquet")

    # Load from cache and sample
    df2 = load_ngram_df(cache_path, format="parquet")
    print(sample_docs_containing(df2, "throatox", 3))
    sys.exit(0)

    # --- Below here unchanged logic that assumes count_docs_containing / sampling still works ---
=======
    print(sample_docs_containing(df, "throatox", 3))
    sys.exit(0)
    # Load expansions and derive the seed vocabulary
>>>>>>> Stashed changes
    all_expansions = load_json('testing/ngram_evals_test_no_digits/4/topk_25_min_cos_0.4_cbow.json')
    global_seed_vocab: List[str] = (
        sorted([str(s) for s, v in all_expansions.items() if isinstance(v, list) and v])
    )

    total = len(global_seed_vocab)
    failures = []
    for seed in global_seed_vocab:
        cnt = count_docs_containing(df2, seed)
        if cnt < 3:
            failures.append((seed, cnt))
            print(f"[MISS] seed='{seed}'  count={cnt}  (< 3)")
        else:
            print(f"[OK]   seed='{seed}'  count={cnt}")

    print("\nSummary:")
    print(f"  Seeds checked: {total}")
    print(f"  Seeds meeting ≥3: {total - len(failures)}")
    print(f"  Seeds failing: {len(failures)}")

    if failures:
        print("\nFailing seeds (seed, count):")
        for s, c in failures:
            print(f"  {s}\t{c}")

    from collections import Counter
    ngram_df_counts = Counter()
    for s in df2["ngram_token_set"]:
        ngram_df_counts.update(s)

    global_expansion_vocab: List[str] = sorted({
        str(term)
        for v in all_expansions.values()
        if isinstance(v, list)
        for term in v
        if isinstance(term, str) and term.strip()
    })

    exp_failures = []
    for term in global_expansion_vocab:
        cnt = ngram_df_counts.get(term, 0)
        if cnt < 3:
            exp_failures.append((term, cnt))
            print(f"[MISS] expansion='{term}'  count={cnt}  (< 3)")
        else:
            print(f"[OK]   expansion='{term}'  count={cnt}")

    print("\nExpansion summary:")
    print(f"  Unique expansions checked: {len(global_expansion_vocab)}")
    print(f"  Expansions meeting ≥3: {len(global_expansion_vocab) - len(exp_failures)}")
    print(f"  Expansions failing: {len(exp_failures)}")

    if exp_failures:
        print("\nFailing expansions (term, count):")
        for t, c in exp_failures:
            print(f"  {t}\t{c}")
