# build_docs_dataframe.py
from __future__ import annotations

"""
Build a document-level DataFrame from a MongoDB collection of Reddit posts/comments.

Features
- Optional `subreddits` filter merged with an arbitrary `query` via $and
- Works in two modes:
    * RAW-ONLY (preprocess=None): metadata + raw text columns (title/selftext/body)
    * PROCESSED (preprocess callable): adds per-field processed text and, if `vocabulary`
      is provided, applies category-aware term matching with counts and frequencies
- Post vs comment handling:
    * Posts: `title`, `selftext` (no `body`)
    * Comments: `body` (no `title`/`selftext`)
- Arrow-friendly string casting for ID-like columns to avoid pyarrow type errors

Returns
- (df: pd.DataFrame, summary: dict)

Notes
- `preprocess` must return List[str] if provided
- `vocabulary` format: {category_name: [term, term, ...], ...}
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from pymongo import MongoClient


def build_docs_dataframe(
    *,
    mongo_uri: str = "mongodb://localhost:27017/",
    db_name: str = "reddit",
    collection_name: str = "noburp_all",
    query: Optional[dict] = None,
    projection: Optional[dict] = None,
    subreddits: Optional[List[str]] = None,
    preprocess: Optional[Callable[[str], List[str]]] = None,  # optional; raw-only if None
    vocabulary: Optional[Dict[str, List[str]]] = None,        # optional; used only if preprocess provided
    timezone_str: str = "America/Chicago",
    pipeline_version: str = "v1",
    limit: Optional[int] = None,
    store_tokens: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a document-level DataFrame from a Mongo collection of Reddit posts/comments.

    Behavior:
    - If `preprocess` is None: returns raw-only schema (no processed fields, no vocab outputs).
    - If `preprocess` is provided: also returns processed text columns and (if `vocabulary` is
      provided) vocabulary matches, frequencies, and category counts.
    - If `subreddits` is provided: merged with `query` via $and to filter by subreddit
      (accepts "noburp" or "r/noburp", case-insensitive).
    """

    # ------------------------ helpers ------------------------
    def _norm(s: str) -> str:
        return s.strip().lower()

    def _as_str(x):
        return None if x is None else str(x)

    def _as_timestamp(x):
        """Coerce to float seconds since epoch or None."""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip():
            try:
                return float(x)
            except Exception:
                return None
        return None

    create_processed = callable(preprocess)
    apply_vocab = create_processed and isinstance(vocabulary, dict) and len(vocabulary) > 0

    # --------- Normalize vocabulary (only if needed) ----------
    if apply_vocab:
        cat_to_terms = {
            cat: {_norm(t) for t in terms if isinstance(t, str) and t.strip()}
            for cat, terms in vocabulary.items()
        }
        term_to_cats: Dict[str, set] = defaultdict(set)
        for cat, terms in cat_to_terms.items():
            for t in terms:
                term_to_cats[t].add(cat)
        all_terms = set(term_to_cats.keys())
    else:
        cat_to_terms, term_to_cats, all_terms = {}, defaultdict(set), set()

    # --------- Build effective Mongo query ----------
    eff_query: dict = (query or {}).copy()

    if subreddits:
        raw = [s for s in subreddits if isinstance(s, str) and s.strip()]
        stripped = [s[2:] if s.lower().startswith("r/") else s for s in raw]
        lowered = [s.lower() for s in stripped]
        # include raw/stripped/lowered to be tolerant; most datasets store lowercase
        subs_all = sorted(set(raw) | set(stripped) | set(lowered))
        sub_filter = {"subreddit": {"$in": subs_all}}

        eff_query = {"$and": [eff_query, sub_filter]} if eff_query else sub_filter

    # --------- Mongo connection ----------
    client = MongoClient(mongo_uri)
    coll = client[db_name][collection_name]

    if projection is None:
        projection = {
            "_id": 1, "name": 1, "id": 1,
            "link_id": 1, "parent_id": 1,
            "author": 1, "author_fullname": 1, "author_created_utc": 1,
            "subreddit": 1, "permalink": 1,
            "created_utc": 1, "created": 1,
            "score": 1, "all_awardings": 1, "gilded": 1,
            "title": 1, "selftext": 1, "body": 1,
            "_meta": 1, "removed_by_category": 1, "is_self": 1, "is_submitter": 1,
        }

    cursor = coll.find(eff_query, projection, no_cursor_timeout=True).batch_size(2000)

    tz_local = ZoneInfo(timezone_str)

    rows = []
    n_seen = n_posts = n_comments = 0
    docs_with_matches = 0

    for doc in cursor:
        if limit is not None and n_seen >= limit:
            break
        n_seen += 1

        # ---------- identify post vs comment ----------
        name = doc.get("name")  # e.g., t1_* or t3_*
        _id = str(doc.get("_id"))
        is_post = bool(name and name.startswith("t3_")) or ("title" in doc or "selftext" in doc)

        n_posts += int(is_post)
        n_comments += int(not is_post)

        # ---------- stable identifiers ----------
        doc_id = name or f"{'t3' if is_post else 't1'}_{doc.get('id', _id)}"
        post_id = _as_str(doc.get("name")) if is_post else _as_str(doc.get("link_id"))
        parent_id = _as_str(doc.get("parent_id"))

        # ---------- metadata ----------
        subreddit = _as_str(doc.get("subreddit"))
        permalink = _as_str(doc.get("permalink"))

        author = _as_str(doc.get("author"))
        author_id = _as_str(doc.get("author_fullname"))
        author_created_utc = _as_timestamp(doc.get("author_created_utc"))

        created_utc = _as_timestamp(doc.get("created_utc")) or _as_timestamp(doc.get("created"))
        created_dt_utc = (datetime.fromtimestamp(created_utc, tz=timezone.utc)
                          if isinstance(created_utc, (int, float, complex)) and created_utc is not None
                          else None)
        created_dt_local = created_dt_utc.astimezone(tz_local) if created_dt_utc else None

        account_age_days = ((created_utc - author_created_utc) / 86400.0
                            if (isinstance(created_utc, (int, float)) and
                                isinstance(author_created_utc, (int, float))) else None)

        score = doc.get("score")
        num_awards = len(doc.get("all_awardings", []) or [])

        # ---------- raw text fields ----------
        title_raw = doc.get("title") if is_post else None
        selftext_raw = doc.get("selftext") if is_post else None
        body_raw = None if is_post else doc.get("body")

        # ---------- base row (raw-only always present) ----------
        row = {
            "doc_id": doc_id,
            "post_id": post_id,
            "parent_id": parent_id,
            "is_post": bool(is_post),

            "author": author,
            "author_id": author_id,
            "author_created_utc": author_created_utc,
            "account_age_days": account_age_days,
            "author_is_deleted": bool(author == "[deleted]"),

            "subreddit": subreddit,
            "permalink": permalink,

            "created_utc": created_utc,
            "created_dt_utc": created_dt_utc,
            "created_dt_local": created_dt_local,
            "iso_year": None,
            "iso_week": None,
            "iso_dow": None,
            "hour": None,

            "score": score,
            "num_awards": num_awards,

            # raw text (empty strings if missing)
            "title": (title_raw or ""),
            "selftext": (selftext_raw or ""),
            "body": (body_raw or ""),

            # provenance
            "pipeline_version": pipeline_version,
            "source_collection": collection_name,
            "was_deleted_later": bool((doc.get("_meta") or {}).get("was_deleted_later", False)),
            "removed_by_category": _as_str(doc.get("removed_by_category")),
        }

        # ---------- convenience time parts ----------
        if created_dt_local:
            iso = created_dt_local.isocalendar()
            row["iso_year"] = iso.year
            row["iso_week"] = iso.week
            row["iso_dow"] = iso.weekday
            row["hour"] = created_dt_local.hour

        # ---------- optional: processed fields + vocab ----------
        if create_processed:
            def _safe_proc(text: Optional[str]) -> List[str]:
                if isinstance(text, str) and text.strip():
                    toks = preprocess(text)  # type: ignore[arg-type]
                    if not isinstance(toks, list):
                        raise TypeError("`preprocess` must return List[str].")
                    return toks
                return []

            title_toks = _safe_proc(title_raw) if is_post else []
            selftext_toks = _safe_proc(selftext_raw) if is_post else []
            body_toks = _safe_proc(body_raw) if not is_post else []

            row.update({
                "title_processed": " ".join(title_toks) if title_toks else "",
                "selftext_processed": " ".join(selftext_toks) if selftext_toks else "",
                "body_processed": " ".join(body_toks) if body_toks else "",
                "title_token_len": len(title_toks),
                "selftext_token_len": len(selftext_toks),
                "body_token_len": len(body_toks),
                "total_token_len": len(title_toks) + len(selftext_toks) + len(body_toks),
            })

            if store_tokens:
                row["title_tokens"] = title_toks
                row["selftext_tokens"] = selftext_toks
                row["body_tokens"] = body_toks

            if apply_vocab:
                toks_norm = [_norm(t) for t in (title_toks + selftext_toks + body_toks)]
                term_counts = Counter(t for t in toks_norm if t in all_terms)
                n_matched = sum(term_counts.values())
                total_tokens = len(toks_norm) if toks_norm else 0

                term_freqs = {t: (term_counts[t] / total_tokens) if total_tokens else 0.0
                              for t in term_counts}
                term_freqs_per_1k = {t: 1000.0 * f for t, f in term_freqs.items()}

                cats_present = sorted({c for t in term_counts for c in term_to_cats.get(t, ())})
                cat_counts = defaultdict(int)
                for t, cnt in term_counts.items():
                    for c in term_to_cats[t]:
                        cat_counts[c] += cnt

                docs_with_matches += int(bool(term_counts))

                row.update({
                    "matched_terms": sorted(term_counts.keys()),
                    "matched_term_counts": dict(term_counts),
                    "matched_term_freqs": term_freqs,
                    "matched_term_freqs_per_1k": term_freqs_per_1k,
                    "unique_matched_terms": len(term_counts),
                    "n_matched_terms": n_matched,
                    "coverage_ratio": (n_matched / total_tokens) if total_tokens else 0.0,
                    "categories_present": cats_present,
                    "cat_counts": dict(cat_counts),
                })

        # ---------- lightweight consistency notes ----------
        if is_post and body_raw:
            row["_anomaly_note"] = "post_with_body_present"
        elif (not is_post) and (title_raw or selftext_raw):
            row["_anomaly_note"] = "comment_with_title_or_selftext"

        rows.append(row)

    # finalize
    df = pd.DataFrame.from_records(rows).convert_dtypes()

    # Arrow-friendly casting for string-ish columns to avoid "Expected bytes, got 'Int64'" errors
    STRING_COLS = [
        "doc_id", "post_id", "parent_id",
        "author", "author_id",
        "subreddit", "permalink",
        "title", "selftext", "body",
        "removed_by_category", "source_collection",
        "pipeline_version", "_anomaly_note",
    ]
    for c in STRING_COLS:
        if c in df.columns:
            # Coerce to pandas string dtype (pyarrow-backed if available)
            try:
                df[c] = df[c].astype("string[pyarrow]")
            except TypeError:
                df[c] = df[c].astype("string")

    summary = {
        "n_docs": len(df),
        "n_posts": n_posts,
        "n_comments": n_comments,
        "schema_mode": "processed" if create_processed else "raw",
        "vocab_applied": bool(apply_vocab),
        "docs_with_matches": docs_with_matches if apply_vocab else 0,
        "vocab_size_terms": (len(all_terms) if apply_vocab else 0),
        "vocab_size_categories": (len(cat_to_terms) if apply_vocab else 0),
        "collection": collection_name,
        "pipeline_version": pipeline_version,
        "subreddits_filter": subreddits or [],
        "effective_query": eff_query,
        "note": ("vocabulary ignored because preprocess=None"
                 if (vocabulary and not create_processed) else ""),
    }
    return df, summary


# ---------------------
# Example usage (RAW-ONLY, no preprocessing yet):
# from pathlib import Path
# df_raw, info_raw = build_docs_dataframe(
#     mongo_uri="mongodb://localhost:27017/",
#     db_name="reddit",
#     collection_name="noburp_all",
#     subreddits=["noburp"],      # optional
#     preprocess=None,            # <- skip processed text
#     vocabulary=None,            # <- vocab is ignored without preprocess
#     limit=30000,                # smoke test
# )
# df_raw.to_parquet("docs_raw.parquet", index=False, compression="zstd")
# print(info_raw)
#
# Example usage (WITH processing + vocabulary):
# from utils.text_pipeline import process_text
# import json
# vocab = json.loads(Path("vocabulary.json").read_text())
# df_proc, info_proc = build_docs_dataframe(
#     mongo_uri="mongodb://localhost:27017/",
#     db_name="reddit",
#     collection_name="noburp_all",
#     subreddits=["noburp"],
#     preprocess=process_text,
#     vocabulary=vocab,
#     pipeline_version="prep-2025-08-20",
# )
# df_proc.to_parquet("docs_proc.parquet", index=False, compression="zstd")
# print(info_proc)


from pathlib import Path
from datetime import datetime
import pandas as pd


OUTDIR = Path("testing/dataframe_visualization"); 
OUTDIR.mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
parquet_path = OUTDIR / f"docs_raw_{ts}.parquet"
csv_path     = OUTDIR / f"docs_raw_{ts}.csv"

# --- Run raw-only (no preprocess, no vocabulary) ---
df, info = build_docs_dataframe(
    mongo_uri="mongodb://localhost:27017/",
    db_name="reddit",
    collection_name="noburp_all",   # change if needed
    subreddits=["noburp"],
    preprocess=None,                # <- raw-only
    vocabulary=None,                # <- ignored when preprocess=None
    limit=30000,                     # smoke test first; remove/raise later
    pipeline_version="raw-boot",
)

print("\n=== SUMMARY ===")
for k, v in info.items():
    print(f"{k}: {v}")

print("\n=== SAMPLE ROWS ===")
print(df.sample(min(5, len(df))).to_string(max_cols=40, max_rows=5))

# Quick sanity checks
print("\n=== QUICK CHECKS ===")
print("is_post value counts:\n", df["is_post"].value_counts(dropna=False))
print("Subreddit top 10:\n", df["subreddit"].value_counts().head(10))
print("Time range (local):", df["created_dt_local"].min(), "→", df["created_dt_local"].max())
print("Deleted/removed flags (%):",
      (df["was_deleted_later"].mean() * 100 if "was_deleted_later" in df else 0.0),
      (df["removed_by_category"].notna().mean() * 100))

# Save
print("\nSaving…")
df.to_parquet(parquet_path, index=False, compression="zstd")
# CSV is larger but handy for eyeballing small samples
(df.head(30000)).to_csv(csv_path, index=False)