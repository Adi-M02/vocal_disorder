# analyze_vocab_dir.py
"""
Load previously saved Parquet files (from your out directory) and run common analyses.

USAGE (examples)
----------------
# Basic: point to the directory that contains your parquet files
python analyze_vocab_dir.py /path/to/out_docs

# Include a vocabulary JSON (recommended if we need to rebuild doc_term from the main df)
python analyze_vocab_dir.py /path/to/out_docs --vocab clustering/test_unigram_model/20250821_191655/cluster_terms_minSim_K11.json

# Change time-series freq and number of items printed
python analyze_vocab_dir.py /path/to/out_docs --freq W --topn 15

# Save analysis tables to CSV as well
python analyze_vocab_dir.py /path/to/out_docs --save-csv

WHAT IT DOES
------------
- Loads the latest parquet files in the directory:
    * docs_proc_*.parquet (or docs_raw_*.parquet as fallback)
    * doc_term_*.parquet (if available)
    * term_summary_*.parquet (if available)
    * category_summary_*.parquet (if available)
- If doc_term is missing, it rebuilds a minimal `doc_term` from the main df:
    * Uses matched_term_counts & matched_term_freqs
    * If you pass --vocab, it will attach correct categories per term from the vocab
      (otherwise categories will be left empty if not already present)
- Prints:
    * Summary sizes
    * Top-N terms overall
    * Top-N terms per category (first few categories)
    * Weekly category counts (head)
    * Category co-occurrences (head)
    * Coverage histogram (bin counts)
- Optionally saves the produced analysis tables to CSV alongside the loaded parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
import pandas as pd
import numpy as np


# ---------------------- utilities ----------------------

def _latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

def _glob_latest(dirpath: Path, patterns: List[str]) -> Optional[Path]:
    all_matches: List[Path] = []
    for pat in patterns:
        all_matches.extend(dirpath.glob(pat))
    return _latest_by_mtime(all_matches)

def _maybe_jsonloads(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return None
    return x if isinstance(x, (dict, list)) else None

def _ensure_list(v) -> List:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            out = json.loads(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


# ---------------------- loaders ----------------------

def load_tables_from_dir(dirpath: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load docs, doc_term, term_summary, category_summary (latest files by mtime)."""
    # main docs parquet
    docs_path = _glob_latest(dirpath, ["docs_proc_*.parquet", "docs_raw_*.parquet", "docs_*.parquet"])
    if not docs_path:
        raise FileNotFoundError(f"No docs parquet found in {dirpath} (looked for docs_proc_*.parquet / docs_raw_*.parquet / docs_*.parquet)")

    df = pd.read_parquet(docs_path)

    # tidy tables (optional)
    dt_path = _glob_latest(dirpath, ["doc_term_*.parquet"])
    ts_path = _glob_latest(dirpath, ["term_summary_*.parquet"])
    cs_path = _glob_latest(dirpath, ["category_summary_*.parquet"])

    doc_term = pd.read_parquet(dt_path) if dt_path else None
    term_summary = pd.read_parquet(ts_path) if ts_path else None
    category_summary = pd.read_parquet(cs_path) if cs_path else None

    # category_summary may have "top_terms_json" string column from earlier saver
    if category_summary is not None and "top_terms_json" in category_summary.columns and "top_terms" not in category_summary.columns:
        try:
            category_summary = category_summary.copy()
            category_summary["top_terms"] = category_summary["top_terms_json"].map(_maybe_jsonloads)
        except Exception:
            pass

    return df, doc_term, term_summary, category_summary


def rebuild_doc_term_from_df(df: pd.DataFrame, vocabulary: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """Rebuild tidy doc_term from main df JSON-ified columns. Attach categories via vocab if provided."""
    df2 = df.copy()

    # parse json columns if stringified
    for col in ["matched_term_counts", "matched_term_freqs"]:
        if col in df2.columns:
            df2[col] = df2[col].map(_maybe_jsonloads)

    # Precompute term -> categories if vocab is available
    term_to_cats: Dict[str, List[str]] = {}
    if isinstance(vocabulary, dict):
        for cat, terms in vocabulary.items():
            for t in terms:
                if not isinstance(t, str) or not t.strip():
                    continue
                key = t.strip().lower()
                term_to_cats.setdefault(key, []).append(cat)

    rows = []
    for _, r in df2.iterrows():
        mcounts = r.get("matched_term_counts")
        if not isinstance(mcounts, dict) or not mcounts:
            continue
        mfreqs = r.get("matched_term_freqs") if isinstance(r.get("matched_term_freqs"), dict) else {}
        for term, cnt in mcounts.items():
            term_key = term.strip().lower() if isinstance(term, str) else str(term)
            cats = term_to_cats.get(term_key, [])
            rows.append({
                "doc_id": r.get("doc_id"),
                "term": term_key,
                "count": int(cnt),
                "freq": float(mfreqs.get(term, np.nan)) if isinstance(mfreqs, dict) else np.nan,
                "categories": cats,  # may be [] if no vocab provided
                "subreddit": r.get("subreddit"),
                "is_post": bool(r.get("is_post")),
                "created_dt_local": r.get("created_dt_local"),
            })
    doc_term = pd.DataFrame(rows)
    return doc_term


# ---------------------- analyses ----------------------

def top_terms_overall(term_summary: pd.DataFrame, n=20) -> pd.DataFrame:
    return term_summary.sort_values(["total_count", "n_docs"], ascending=[False, False]).head(n)

def top_terms_by_category(doc_term: pd.DataFrame, n=20) -> pd.DataFrame:
    dt = doc_term.copy()
    # categories may be a JSON string or list
    dt["categories"] = dt["categories"].map(_ensure_list)
    dt = dt.explode("categories").dropna(subset=["categories"])
    g = (dt.groupby(["categories","term"], as_index=False)
           .agg(total_count=("count","sum"), n_docs=("doc_id","nunique")))
    g["rank"] = g.groupby("categories")["total_count"].rank(method="first", ascending=False)
    return g[g["rank"] <= n].sort_values(["categories","total_count"], ascending=[True, False])

def category_timeseries(doc_term: pd.DataFrame, freq="W") -> pd.DataFrame:
    dt = doc_term.copy()
    dt["categories"] = dt["categories"].map(_ensure_list)
    dt = dt.explode("categories").dropna(subset=["categories"])
    dt["created_dt_local"] = pd.to_datetime(dt["created_dt_local"], utc=True, errors="coerce")
    ts = (dt.set_index("created_dt_local")
            .groupby([pd.Grouper(freq=freq), "categories"])["count"].sum()
            .reset_index())
    return ts.rename(columns={"created_dt_local": "date", "count": "count"})

def category_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    # categories_present may be a list or a JSON string of categories present per *document*.
    def to_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                out = json.loads(v)
                return out if isinstance(out, list) else []
            except Exception:
                return []
        return []
    pairs = {}
    for cats in df.get("categories_present", pd.Series([], dtype=object)).map(to_list):
        uniq = sorted(set(cats))
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                key = (uniq[i], uniq[j])
                pairs[key] = pairs.get(key, 0) + 1
    co = pd.DataFrame([(a,b,c) for (a,b), c in pairs.items()], columns=["cat_a","cat_b","n_docs"])
    return co.sort_values("n_docs", ascending=False)

def coverage_histogram(df: pd.DataFrame, bins=20) -> pd.Series:
    s = pd.to_numeric(df.get("coverage_ratio"), errors="coerce").dropna()
    return s.value_counts(bins=bins).sort_index()

def build_term_summary_from_doc_term(doc_term: pd.DataFrame) -> pd.DataFrame:
    g = (doc_term.groupby("term", as_index=False)
            .agg(total_count=("count","sum"),
                 n_docs=("doc_id","nunique"),
                 mean_freq=("freq","mean")))
    # categories per term (union)
    cats_map = (doc_term.explode("categories")
                        .dropna(subset=["categories"])
                        .groupby("term")["categories"]
                        .agg(lambda x: sorted(set(_ensure_list(x.tolist()))))
                        .to_dict())
    g["categories"] = g["term"].map(lambda t: cats_map.get(t, []))
    return g.sort_values(["total_count","n_docs"], ascending=[False, False])


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze vocabulary counts from a directory of Parquet files.")
    ap.add_argument("data_dir", type=str, help="Path to directory containing docs_*.parquet (and optional tidy tables).")
    ap.add_argument("--vocab", type=str, default=None, help="Optional path to vocabulary JSON (used if doc_term needs to be rebuilt).")
    ap.add_argument("--freq", type=str, default="W", help="Pandas offset alias for time series (e.g., 'W', 'M'). Default: W")
    ap.add_argument("--topn", type=int, default=15, help="Top-N items to print. Default: 15")
    ap.add_argument("--save-csv", action="store_true", help="Also save the produced analysis tables to CSV in the same directory.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Directory not found: {data_dir}")

    vocab = None
    if args.vocab:
        vp = Path(args.vocab).expanduser().resolve()
        vocab = json.loads(vp.read_text(encoding="utf-8"))

    # Load tables
    df, doc_term, term_summary, category_summary = load_tables_from_dir(data_dir)

    print("\n=== Loaded ===")
    print(f"docs: {len(df)} | doc_term: {len(doc_term) if doc_term is not None else 0} "
          f"| term_summary: {len(term_summary) if term_summary is not None else 0} "
          f"| category_summary: {len(category_summary) if category_summary is not None else 0}")

    # Rebuild doc_term if missing
    if doc_term is None or doc_term.empty:
        print("doc_term parquet not found; rebuilding from main df…")
        doc_term = rebuild_doc_term_from_df(df, vocabulary=vocab)

    # Build term_summary if missing
    if term_summary is None or term_summary.empty:
        print("term_summary parquet not found; building from doc_term…")
        term_summary = build_term_summary_from_doc_term(doc_term)

    # If category_summary missing, we can approximate from doc_term (top terms per category)
    if category_summary is None or category_summary.empty:
        print("category_summary parquet not found; approximating from doc_term…")
        # build a minimal category summary
        dt = doc_term.copy()
        dt["categories"] = dt["categories"].map(_ensure_list)
        ex = dt.explode("categories").dropna(subset=["categories"])
        if len(ex):
            cat_totals = ex.groupby("categories", as_index=False).agg(
                total_count=("count", "sum"),
                n_docs=("doc_id", "nunique"),
            ).rename(columns={"categories": "category"}).sort_values("total_count", ascending=False)
            # create a 'top_terms' preview column
            top_terms = []
            for cat in cat_totals["category"]:
                sub = ex[ex["categories"] == cat]
                tops = (sub.groupby("term", as_index=False)["count"].sum()
                          .sort_values("count", ascending=False)
                          .head(50))
                top_terms.append(list(map(tuple, tops[["term","count"]].to_records(index=False))))
            cat_totals["top_terms"] = top_terms
            category_summary = cat_totals
        else:
            category_summary = pd.DataFrame(columns=["category","total_count","n_docs","top_terms"])

    # --------- Prints ---------
    print("\nTop terms overall:")
    print(top_terms_overall(term_summary, n=args.topn).to_string(index=False))

    if len(doc_term):
        print("\nTop terms per category (preview):")
        topk = top_terms_by_category(doc_term, n=args.topn)
        for cat in topk["categories"].drop_duplicates().head(3):
            print(f"\nCategory: {cat}")
            print(topk[topk["categories"] == cat][["term","total_count","n_docs"]]
                  .head(args.topn).to_string(index=False))

        ts = category_timeseries(doc_term, freq=args.freq)
        print(f"\nCategory time series (freq={args.freq}) head:")
        print(ts.head(12).to_string(index=False))

        co = category_cooccurrence(df)
        print("\nTop category co-occurrences:")
        print(co.head(args.topn).to_string(index=False))

        print("\nCoverage histogram (bin counts):")
        print(coverage_histogram(df, bins=15))

    # --------- Optional CSV outputs ---------
    if args.save_csv:
        ts = category_timeseries(doc_term, freq=args.freq)
        (data_dir / "analysis_outputs").mkdir(exist_ok=True)
        outdir = data_dir / "analysis_outputs"

        doc_term.to_csv(outdir / "doc_term.csv", index=False)
        term_summary.to_csv(outdir / "term_summary.csv", index=False)
        category_summary.to_csv(outdir / "category_summary.csv", index=False)
        ts.to_csv(outdir / "category_timeseries.csv", index=False)

        print(f"\nSaved CSVs to: {outdir}")

if __name__ == "__main__":
    main()
