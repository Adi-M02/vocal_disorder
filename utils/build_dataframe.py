from __future__ import annotations

"""
Build a document-level DataFrame from a MongoDB collection of Reddit posts/comments,
apply a category-structured vocabulary, and produce tidy tables for downstream analysis.

Key features
- Import your processors from your repo path:
    sys.path.append('../vocab_disorder')
    from utils.text_pipeline import process_text, ngram_and_process
- Flexible processing modes:
    1) n-gram processing (ngram_and_process(text, ngram_phraser_dir))
    2) plain process_text(text, **kwargs)
    3) custom preprocess callable
    4) or RAW-ONLY (no processing)
- Vocabulary application scope:
    * auto/post_fields (DEFAULT): posts → title+selftext; comments → body
    * body_only            : use only body (posts usually have empty body)
    * all_fields           : combine title+selftext+body for any doc
- Per-document outputs:
    * raw text fields (title/selftext/body)
    * optional processed fields per text field (+ token lengths)
    * matched_terms, matched_term_counts, matched_term_freqs, categories_present, cat_counts
- Downstream helpers:
    * make_vocab_tables(...) → (doc_term, term_summary, category_summary)

Notes
- Vocabulary format: {category_name: [term, term, ...], ...}
- Processing functions must return List[str] when used
"""

import sys
sys.path.append('../vocab_disorder')  # per your request
from text_pipeline import process_text as _default_process_text, ngram_and_process as _default_ngram_and_process  # noqa: E402

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from pymongo import MongoClient


# ------------------------ Public API ------------------------

def build_docs_dataframe(
    *,
    mongo_uri: str = "mongodb://localhost:27017/",
    db_name: str = "reddit",
    collection_name: str = "noburp_all",
    query: Optional[dict] = None,
    projection: Optional[dict] = None,
    subreddits: Optional[List[str]] = None,

    # Processing choices (pick one path; priority: ngrams > process_text > preprocess > raw-only)
    ngram_phraser_dir: Optional[str] = None,
    ngram_and_process_fn: Optional[Callable[[str, str], List[str]]] = None,
    process_text_fn: Optional[Callable[..., List[str]]] = None,
    process_text_kwargs: Optional[Mapping[str, Any]] = None,
    preprocess: Optional[Callable[[str], List[str]]] = None,

    # Vocabulary + scope
    vocabulary: Optional[Dict[str, List[str]]] = None,
    vocab_scope: str = "post_fields",   # "post_fields"|"auto" (alias), "body_only", "all_fields"

    # Misc
    timezone_str: str = "America/Chicago",
    pipeline_version: str = "v1",
    limit: Optional[int] = None,
    store_tokens: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a document-level DataFrame with raw/processed text and (optionally) vocabulary matches.

    Parameters
    ----------
    subreddits : list[str] | None
        If provided, merged into `query` via $and as {"subreddit": {"$in": ...}}.
        Accepts "r/foo" or "foo" (case-insensitive).
    ngram_phraser_dir, ngram_and_process_fn :
        If `ngram_phraser_dir` is set, uses `ngram_and_process_fn(text, dir)` (or default import).
    process_text_fn, process_text_kwargs :
        Else uses `process_text_fn(text, **kwargs)` (or default import) if provided.
    preprocess :
        Else uses this callable(text) -> List[str].
    vocabulary : dict | None
        {category: [terms...]} matched on processed tokens (normalized lowercase).
    vocab_scope : str
        "post_fields"/"auto" : posts → title+selftext; comments → body (DEFAULT)
        "body_only"          : body only for all docs
        "all_fields"         : title+selftext+body combined for all docs
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

    def _prepare_vocab_maps(vocab: Dict[str, List[str]]):
        cat_to_terms = {
            cat: {_norm(t) for t in terms if isinstance(t, str) and t.strip()}
            for cat, terms in vocab.items()
        }
        term_to_cats: Dict[str, set] = defaultdict(set)
        for cat, terms in cat_to_terms.items():
            for t in terms:
                term_to_cats[t].add(cat)
        all_terms = set(term_to_cats.keys())
        return cat_to_terms, term_to_cats, all_terms

    def _tokens_for_scope(
        is_post: bool,
        title_toks: List[str],
        selftext_toks: List[str],
        body_toks: List[str],
        scope: str,
    ) -> List[str]:
        if scope in ("post_fields", "auto"):
            return (title_toks + selftext_toks) if is_post else body_toks
        elif scope == "body_only":
            return body_toks
        elif scope == "all_fields":
            return title_toks + selftext_toks + body_toks
        else:
            # fallback to default
            return (title_toks + selftext_toks) if is_post else body_toks

    # Determine effective processing function
    effective_preprocess: Optional[Callable[[str], List[str]]] = None
    processing_mode = "none"

    if ngram_phraser_dir:
        _ng_fn = ngram_and_process_fn or _default_ngram_and_process
        if callable(_ng_fn):
            effective_preprocess = lambda text: _ng_fn(text, ngram_phraser_dir)  # type: ignore[arg-type]
            processing_mode = "ngrams"
    elif process_text_fn or _default_process_text:
        _pt_fn = process_text_fn or _default_process_text
        if callable(_pt_fn):
            kwargs = dict(process_text_kwargs or {})
            effective_preprocess = lambda text: _pt_fn(text, **kwargs)  # type: ignore[misc]
            processing_mode = "process_text"
    elif callable(preprocess):
        effective_preprocess = preprocess
        processing_mode = "custom"
    else:
        processing_mode = "none"

    create_processed = effective_preprocess is not None
    apply_vocab = create_processed and isinstance(vocabulary, dict) and len(vocabulary) > 0

    # --------- Normalize vocabulary (only if needed) ----------
    if apply_vocab:
        cat_to_terms, term_to_cats, all_terms = _prepare_vocab_maps(vocabulary)  # type: ignore[arg-type]
    else:
        cat_to_terms, term_to_cats, all_terms = {}, defaultdict(set), set()

    # --------- Build effective Mongo query ----------
    eff_query: dict = (query or {}).copy()
    if subreddits:
        raw = [s for s in subreddits if isinstance(s, str) and s.strip()]
        stripped = [s[2:] if s.lower().startswith("r/") else s for s in raw]
        lowered = [s.lower() for s in stripped]
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
                          if isinstance(created_utc, (int, float)) and created_utc is not None
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
                    toks = effective_preprocess(text)  # type: ignore[operator]
                    if not isinstance(toks, list):
                        raise TypeError("processing function must return List[str].")
                    return toks
                return []

            title_toks = _safe_proc(title_raw) if is_post else []
            selftext_toks = _safe_proc(selftext_raw) if is_post else []
            body_toks = _safe_proc(body_raw) if not is_post else []

            row.update({
                # store processed text per field (space-joined for readability)
                "title_processed": " ".join(title_toks) if title_toks else "",
                "selftext_processed": " ".join(selftext_toks) if selftext_toks else "",
                "body_processed": " ".join(body_toks) if body_toks else "",

                # lengths
                "title_token_len": len(title_toks),
                "selftext_token_len": len(selftext_toks),
                "body_token_len": len(body_toks),
                "total_token_len": len(title_toks) + len(selftext_toks) + len(body_toks),

                # processing metadata
                "processing_mode": processing_mode,
                "ngram_phraser_dir": ngram_phraser_dir if processing_mode == "ngrams" else None,
                "vocab_scope": vocab_scope,
            })

            if store_tokens:
                row["title_tokens"] = title_toks
                row["selftext_tokens"] = selftext_toks
                row["body_tokens"] = body_toks

            # ----- vocabulary application (according to scope) -----
            if apply_vocab:
                toks_for_vocab = _tokens_for_scope(
                    is_post, title_toks, selftext_toks, body_toks, vocab_scope
                )
                toks_norm = [_norm(t) for t in toks_for_vocab]
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

    # Arrow-friendly casting for string-ish columns
    STRING_COLS = [
        "doc_id", "post_id", "parent_id",
        "author", "author_id",
        "subreddit", "permalink",
        "title", "selftext", "body",
        "removed_by_category", "source_collection",
        "pipeline_version", "_anomaly_note",
        "processing_mode", "ngram_phraser_dir", "vocab_scope",
    ]
    for c in STRING_COLS:
        if c in df.columns:
            try:
                df[c] = df[c].astype("string[pyarrow]")
            except TypeError:
                df[c] = df[c].astype("string")

    summary = {
        "n_docs": len(df),
        "n_posts": n_posts,
        "n_comments": n_comments,
        "schema_mode": ("processed" if create_processed else "raw"),
        "processing_mode": processing_mode,
        "vocab_applied": bool(apply_vocab),
        "vocab_scope": vocab_scope,
        "docs_with_matches": docs_with_matches if apply_vocab else 0,
        "vocab_size_terms": (len(all_terms) if apply_vocab else 0),
        "vocab_size_categories": (len(cat_to_terms) if apply_vocab else 0),
        "collection": collection_name,
        "pipeline_version": pipeline_version,
        "subreddits_filter": subreddits or [],
        "effective_query": eff_query,
        "process_text_kwargs": dict(process_text_kwargs or {}) if processing_mode == "process_text" else {},
        "note": ("vocabulary ignored because no processing function was set"
                 if (vocabulary and not create_processed) else ""),
    }
    return df, summary


def make_vocab_tables(
    df: pd.DataFrame,
    vocabulary: Dict[str, List[str]],
    *,
    # When df has processed lengths, use them to compute frequencies where missing
    default_total_tokens_col: str = "total_token_len",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build tidy tables for downstream analyses:
      1) doc_term:         (doc_id, term, count, freq, categories[list], subreddit, is_post, created_dt_local, ...)
      2) term_summary:     (term, total_count, n_docs, mean_freq, categories[list])
      3) category_summary: (category, total_count, n_docs, top_terms[list of (term,count)])

    Frequencies are per-document (count / total_tokens). If `freq` already exists in df's
    matched_term_freqs for a given term, we reuse that; otherwise we compute from total tokens.

    Note: terms can belong to multiple categories; we keep categories as a list per term/doc.
    """
    def _norm(s: str) -> str:
        return s.strip().lower()

    # maps
    cat_to_terms = {cat: { _norm(t) for t in terms if isinstance(t, str) and t.strip()}
                    for cat, terms in vocabulary.items()}
    term_to_cats: Dict[str, set] = defaultdict(set)
    for cat, terms in cat_to_terms.items():
        for t in terms:
            term_to_cats[t].add(cat)

    # 1) doc_term long table
    rows: List[Dict[str, Any]] = []
    m_counts_col = "matched_term_counts"
    m_freqs_col = "matched_term_freqs"

    has_counts = m_counts_col in df.columns
    has_freqs = m_freqs_col in df.columns
    total_col = default_total_tokens_col if default_total_tokens_col in df.columns else None

    for _, r in df.iterrows():
        doc_id = r.get("doc_id")
        subreddit = r.get("subreddit")
        is_post = r.get("is_post")
        created_dt_local = r.get("created_dt_local")
        total_tokens = r.get(total_col) if total_col else None
        term_counts = r.get(m_counts_col) if has_counts else None
        term_freqs = r.get(m_freqs_col) if has_freqs else {}

        if not isinstance(term_counts, dict) or not term_counts:
            continue

        for term, cnt in term_counts.items():
            cats = sorted(term_to_cats.get(term, []))
            # freq: prefer stored, else compute
            freq = term_freqs.get(term) if isinstance(term_freqs, dict) else None
            if freq is None and isinstance(total_tokens, (int, float)) and total_tokens:
                freq = float(cnt) / float(total_tokens)
            rows.append({
                "doc_id": doc_id,
                "term": term,
                "count": int(cnt),
                "freq": float(freq) if freq is not None else None,
                "categories": cats,
                "subreddit": subreddit,
                "is_post": bool(is_post),
                "created_dt_local": created_dt_local,
            })

    doc_term = pd.DataFrame(rows)
    if len(doc_term):
        try:
            doc_term["doc_id"] = doc_term["doc_id"].astype("string[pyarrow]")
            doc_term["term"] = doc_term["term"].astype("string[pyarrow]")
        except TypeError:
            doc_term["doc_id"] = doc_term["doc_id"].astype("string")
            doc_term["term"] = doc_term["term"].astype("string")

    # 2) term_summary
    if len(doc_term):
        g = doc_term.groupby("term", as_index=False).agg(
            total_count=("count", "sum"),
            n_docs=("doc_id", "nunique"),
            mean_freq=("freq", "mean"),
        )
        # categories per term (globally)
        cats_map = {t: sorted(term_to_cats.get(t, [])) for t in g["term"]}
        g["categories"] = g["term"].map(cats_map)
        term_summary = g.sort_values(["total_count", "n_docs"], ascending=[False, False], kind="mergesort")
    else:
        term_summary = pd.DataFrame(columns=["term", "total_count", "n_docs", "mean_freq", "categories"])

    # 3) category_summary
    # explode categories to count per-category totals and doc coverage
    if len(doc_term):
        expl = doc_term.explode("categories")
        expl = expl[~expl["categories"].isna()]
        cat_totals = expl.groupby("categories", as_index=False).agg(
            total_count=("count", "sum"),
            n_docs=("doc_id", "nunique"),
        )
        # Top terms per category
        top_terms_rows: List[Dict[str, Any]] = []
        for cat in cat_totals["categories"]:
            sub = expl.loc[expl["categories"] == cat]
            tops = (sub.groupby("term", as_index=False)["count"].sum()
                      .sort_values("count", ascending=False)
                      .head(50))
            top_terms_rows.append({"category": cat, "top_terms": list(map(tuple, tops[["term", "count"]].to_records(index=False)))})
        top_terms_df = pd.DataFrame(top_terms_rows)
        category_summary = cat_totals.merge(top_terms_df, left_on="categories", right_on="category", how="left")
        category_summary = category_summary.drop(columns=["category"]).rename(columns={"categories": "category"})
        category_summary = category_summary.sort_values("total_count", ascending=False)
    else:
        category_summary = pd.DataFrame(columns=["category", "total_count", "n_docs", "top_terms"])

    return doc_term, term_summary, category_summary


# ------------------------ Examples ------------------------
# Raw-only (build baseline schema)
# df_raw, info_raw = build_docs_dataframe(
#     mongo_uri="mongodb://localhost:27017/",
#     db_name="reddit",
#     collection_name="noburp_all",
#     subreddits=["noburp"],
# )
# df_raw.to_parquet("docs_raw.parquet", index=False, compression="zstd")
# print(info_raw)
#
# With processing + vocabulary (default: posts use title+selftext; comments use body)
# import json
# vocab = json.load(open("vocabulary.json"))
# df_proc, info_proc = build_docs_dataframe(
#     subreddits=["noburp"],
#     process_text_kwargs={"stoplist": True, "lemmatize": True, "lookup_path": "testing/combined_lemmas.json"},
#     vocabulary=vocab,
#     vocab_scope="post_fields",  # or "body_only" or "all_fields"
#     pipeline_version="proc-2025-08-21",
# )
# df_proc.to_parquet("docs_proc.parquet", index=False, compression="zstd")
# print(info_proc)
#
# Downstream: make tidy tables and get "most common terms per category"
# doc_term, term_summary, category_summary = make_vocab_tables(df_proc, vocab)
# # Examples:
# # Top terms (overall):
# print(term_summary.head(20))
# # Top categories and their top 50 terms:
# print(category_summary.head(10))

from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# ---------- Config ----------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "reddit"
COLLECTION = "noburp_all"
SUBREDDITS = ["noburp"]            # add more subs if needed (or set to None to include all)

# Use default process_text imported inside build_docs_dataframe
PROCESS_TEXT_KW = {
    "stoplist": True,
    "lemmatize": True,
    # "lookup_path": "testing/combined_lemmas.json",  # uncomment to override default
}

# Vocabulary path you gave
VOCAB_PATH = Path("clustering/test_unigram_model/20250821_191655/cluster_terms_minSim_K11.json")

# Output paths
OUTDIR = Path("out_docs"); OUTDIR.mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
parquet_path   = OUTDIR / f"docs_proc_{ts}.parquet"
csv_path       = OUTDIR / f"docs_proc_sample_{ts}.csv"
doc_term_path  = OUTDIR / f"doc_term_{ts}.parquet"
term_sum_path  = OUTDIR / f"term_summary_{ts}.parquet"
cat_sum_path   = OUTDIR / f"category_summary_{ts}.parquet"

# >>>>>>> Analyze the ENTIRE dataset (no sampling) <<<<<<<
LIMIT = None  # <- IMPORTANT: process every document returned by your query

# ---------- Helpers ----------
JSONIFY_COLS_MAIN = [
    # Lists/dicts that Parquet struggles with unless explicitly typed
    "matched_terms",
    "matched_term_counts",
    "matched_term_freqs",
    "matched_term_freqs_per_1k",
    "categories_present",
    "cat_counts",
    "title_tokens",
    "selftext_tokens",
    "body_tokens",
]

def jsonify_col(df: pd.DataFrame, col: str) -> None:
    """Convert list/dict/tuple objects to JSON strings for Parquet safety."""
    if col not in df.columns:
        return
    def _to_json(x):
        if isinstance(x, (list, dict, tuple)):
            try:
                return json.dumps(x, ensure_ascii=False, default=lambda o: str(o))
            except TypeError:
                return json.dumps(str(x), ensure_ascii=False)
        return x
    df[col] = df[col].map(_to_json)
    # Cast to pandas string so pyarrow treats it as utf8
    try:
        df[col] = df[col].astype("string[pyarrow]")
    except TypeError:
        df[col] = df[col].astype("string")

def jsonify_top_terms(category_summary: pd.DataFrame) -> pd.DataFrame:
    """Make category_summary.top_terms Arrow-safe by converting to JSON text."""
    if "top_terms" not in category_summary.columns:
        return category_summary
    def _tt_to_json(lst):
        if not isinstance(lst, list):
            return "[]"
        out = []
        for item in lst:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                term, cnt = item
                try:
                    cnt = int(cnt)
                except Exception:
                    pass
                out.append({"term": str(term), "count": cnt})
            elif isinstance(item, dict):
                term = item.get("term")
                cnt  = item.get("count")
                try:
                    cnt = int(cnt) if cnt is not None else None
                except Exception:
                    pass
                if term is not None:
                    out.append({"term": str(term), "count": cnt})
        return json.dumps(out, ensure_ascii=False)
    category_summary = category_summary.copy()
    category_summary["top_terms_json"] = category_summary["top_terms"].map(_tt_to_json)
    category_summary = category_summary.drop(columns=["top_terms"])
    try:
        category_summary["top_terms_json"] = category_summary["top_terms_json"].astype("string[pyarrow]")
        category_summary["category"] = category_summary["category"].astype("string[pyarrow]")
    except TypeError:
        category_summary["top_terms_json"] = category_summary["top_terms_json"].astype("string")
        category_summary["category"] = category_summary["category"].astype("string")
    return category_summary

# ---------- Load vocabulary ----------
vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

df, info = build_docs_dataframe(
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    collection_name=COLLECTION,
    subreddits=SUBREDDITS,                   # set to None to include all subs in the collection
    process_text_kwargs=PROCESS_TEXT_KW,     # use default process_text with these kwargs
    vocabulary=vocab,
    vocab_scope="post_fields",               # posts: title+selftext, comments: body
    limit=LIMIT,                             # <- None = ENTIRE DATASET
    pipeline_version="proc-defaultPT-vocab",
    store_tokens=False,
)

# ---------- Summary + sanity prints ----------
print("\n=== SUMMARY ===")
for k, v in info.items():
    print(f"{k}: {v}")

print("\n=== SAMPLE ROWS ===")
print(df.sample(min(5, len(df))).to_string(max_cols=40, max_rows=5))

print("\n=== QUICK CHECKS ===")
print("is_post value counts:\n", df["is_post"].value_counts(dropna=False))
print("Subreddit top 10:\n", df["subreddit"].value_counts().head(10))
print("Time range (local):", df["created_dt_local"].min(), "→", df["created_dt_local"].max())
print("Deleted/removed flags (%):",
      (df["was_deleted_later"].mean() * 100 if "was_deleted_later" in df else 0.0),
      (df["removed_by_category"].notna().mean() * 100))

# ---------- Arrow-safe conversions for complex columns on the main DF ----------
for col in JSONIFY_COLS_MAIN:
    jsonify_col(df, col)

# ---------- Save the main DataFrame ----------
print("\nSaving main DataFrame…")
df.to_parquet(parquet_path, index=False, compression="zstd")
# Also write a small CSV preview (keep small even on full runs)
df.head(10000).to_csv(csv_path, index=False)
print(f"Wrote: {parquet_path}")
print(f"Wrote preview CSV (10k rows): {csv_path}")

# ---------- Build tidy tables for analyses over the FULL dataset ----------
print("\nBuilding tidy vocab tables…")
doc_term, term_summary, category_summary = make_vocab_tables(df, vocab)

# Arrow-safe tweaks for tidy tables
if "categories" in doc_term.columns:
    jsonify_col(doc_term, "categories")  # list -> json string

category_summary = jsonify_top_terms(category_summary)

# Save tidy tables (Parquet)
doc_term.to_parquet(doc_term_path, index=False, compression="zstd")
term_summary.to_parquet(term_sum_path, index=False, compression="zstd")
category_summary.to_parquet(cat_sum_path, index=False, compression="zstd")
print(f"Wrote: {doc_term_path}")
print(f"Wrote: {term_sum_path}")
print(f"Wrote: {cat_sum_path}")

# ---------- Quick analytics preview on the FULL dataset ----------
if len(term_summary):
    print("\nTop 15 terms overall by total_count:")
    print(term_summary.head(15).to_string(index=False))
if len(category_summary):
    print("\nTop 10 categories by total_count:")
    print(category_summary[["category", "total_count", "n_docs"]].head(10).to_string(index=False))

    print("\nTop terms (JSON) for first 3 categories:")
    for cat in category_summary["category"].head(3):
        tops_json = category_summary.loc[category_summary["category"] == cat, "top_terms_json"].iloc[0]
        print(f"- {cat}: {tops_json[:200]}{'...' if len(tops_json) > 200 else ''}")
