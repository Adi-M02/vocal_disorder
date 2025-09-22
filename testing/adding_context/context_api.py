#!/usr/bin/env python3
"""
context_api.py — public library API + CLI for querying **base-text** windows
from a cached corpus, using the enhanced behavior:
  - windows centered on the term’s base span,
  - ranked by closeness to target length (2*window + len(term_in_base_tokens)),
  - minimal prettify (HTML unescape, zero-width removal, markdown de-escape).

Usage (CLI):
  python context_api.py --cache cache/ngram_df_baseTokens_and_ngramText.parquet \
                        --term botox_procedure \
                        -k 5 \
                        --window 5 \
                        --format parquet \
                        --module test_get_similar \
                        --json
"""

from typing import List, Dict, Sequence
import importlib
import argparse
import json
import pandas as pd

# Change to your actual setup module filename (without .py)
# It must export:
#   - load_ngram_df(cache_path, format="parquet") -> pd.DataFrame
#   - sample_base_windows_around_term(df, term, k, window) -> List[List[str]]
DEFAULT_SETUP_MODULE = "test_get_similar"

__all__ = [
    "load_cached_corpus",
    "query_base_windows",
    "query_term_windows",               # backward-compat alias
    "query_from_cache",
    "query_base_windows_for_terms",     # NEW (multi-term)
    "query_from_cache_for_terms",       # NEW (multi-term + load)
]


def _load_setup_module(module_name: str):
    return importlib.import_module(module_name)


def load_cached_corpus(
    cache_path: str,
    *,
    format: str = "parquet",
    setup_module: str = DEFAULT_SETUP_MODULE,
) -> pd.DataFrame:
    """Load cached dataframe (doc_id, base_tokens, ngram_text) and rebuild derived cols."""
    mod = _load_setup_module(setup_module)
    return mod.load_ngram_df(cache_path, format=format)


def query_base_windows(
    df: pd.DataFrame,
    term: str,
    k: int,
    window: int,
    *,
    setup_module: str = DEFAULT_SETUP_MODULE,
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

    mod = _load_setup_module(setup_module)
    return mod.sample_base_windows_around_term(df, term, k, window)


# Back-compat alias (old name)
def query_term_windows(
    df: pd.DataFrame,
    term: str,
    k: int,
    window: int,
    *,
    setup_module: str = DEFAULT_SETUP_MODULE,
) -> List[List[str]]:
    """Deprecated: use query_base_windows. Kept for compatibility."""
    return query_base_windows(df, term, k, window, setup_module=setup_module)


def query_from_cache(
    cache_path: str,
    term: str,
    k: int,
    window: int,
    *,
    format: str = "parquet",
    setup_module: str = DEFAULT_SETUP_MODULE,
) -> List[List[str]]:
    """Convenience: load cache and query base windows in one call."""
    df = load_cached_corpus(cache_path, format=format, setup_module=setup_module)
    return query_base_windows(df, term, k, window, setup_module=setup_module)


# ---------- NEW: multi-term helpers ----------

def query_base_windows_for_terms(
    df: pd.DataFrame,
    terms: Sequence[str],
    k: int,
    window: int,
    *,
    setup_module: str = DEFAULT_SETUP_MODULE,
) -> Dict[str, List[List[str]]]:
    """
    Query base windows for multiple terms at once.
    Returns a dict mapping each term -> List[List[str]] (same per-doc windows structure).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if window < 0:
        raise ValueError("window must be >= 0")

    mod = _load_setup_module(setup_module)
    out: Dict[str, List[List[str]]] = {}
    for term in terms:
        out[term] = mod.sample_base_windows_around_term(df, term, k, window)
    return out


def query_from_cache_for_terms(
    cache_path: str,
    terms: Sequence[str],
    k: int,
    window: int,
    *,
    format: str = "parquet",
    setup_module: str = DEFAULT_SETUP_MODULE,
) -> Dict[str, List[List[str]]]:
    """
    Convenience: load cache once, then query windows for multiple terms.
    Returns {term: List[List[str]]}.
    """
    df = load_cached_corpus(cache_path, format=format, setup_module=setup_module)
    return query_base_windows_for_terms(df, terms, k, window, setup_module=setup_module)


# ---------------- CLI ----------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Query **base-text** windows around a term from a cached n-gram dataframe."
    )
    p.add_argument("--cache", default="cache/ngram_df_baseTokens_and_ngramText.parquet",
                   help="Path to cached DF (parquet/feather/pickle).")
    p.add_argument("--term", required=True,
                   help="Exact n-gram token to search (e.g., botox_procedure).")
    p.add_argument("-k", type=int, required=True,
                   help="Max number of matching documents to return.")
    p.add_argument("--window", type=int, required=True,
                   help="Context window size on each side (in base tokens).")
    p.add_argument("--format", default="parquet",
                   choices=["parquet", "feather", "pickle"],
                   help="Cache format.")
    p.add_argument("--module", default=DEFAULT_SETUP_MODULE,
                   help=f"Module name of your setup file (without .py). Default: {DEFAULT_SETUP_MODULE}")
    p.add_argument("--json", action="store_true",
                   help="Print JSON to stdout (default is a readable text format).")
    return p


def main():
    args = _build_argparser().parse_args()
    df = load_cached_corpus(args.cache, format=args.format, setup_module=args.module)
    out = query_base_windows(df, args.term, args.k, args.window, setup_module=args.module)

    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        if not out:
            print("[]")
            return
        for i, winlist in enumerate(out, 1):
            print(f"Doc {i} — {len(winlist)} hits")
            for w in winlist:
                print(f"  - {w}")


if __name__ == "__main__":
    # main()
    print(query_from_cache_for_terms(cache_path="cache/ngram_df_baseTokens_and_ngramText.parquet",
                                      terms=["botox_procedure", "throatox"],
                                      k=3,
                                      window=5))
