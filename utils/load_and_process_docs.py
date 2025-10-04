# parallel_process_noburp.py
import os
import sys
import logging
from typing import Iterator, Any, Dict, List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pymongo
from tqdm import tqdm

sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize
from utils.load_lemmatizer import load_lookup
from spellchecker_folder.spellchecker import spellcheck_token_list
from utils.stopwords import STOPWORDS

# -------- Hardcoded constants --------
DB_NAME = "reddit"
COLLECTION_NAME = "noburp_all"
FILTER_SUBREDDIT = "noburp"
LOOKUP_PATH = "testing/adding_context/combined_lemmas.json"

# -------- Worker globals --------
_LOOKUP: Dict[str, str] | None = None


def _init_worker():
    """Runs once per worker process to load heavy resources."""
    global _LOOKUP
    _LOOKUP = load_lookup(LOOKUP_PATH)

    # Avoid oversubscription for BLAS threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _process_one(text: str, tokenize: bool = True, stoplist: bool = True, lemmatize: bool = True) -> List[str]:
    """Your process_text logic using preloaded lookup."""
    assert _LOOKUP is not None, "Worker not initialized; missing lookup."
    if not tokenize:
        return text.split()

    # First pass tokenize + optional lemmatize
    toks = clean_and_tokenize(text)
    if lemmatize:
        toks = [_LOOKUP.get(tok, tok) for tok in toks]

    # Normalize spacing then retokenize
    text = " ".join(toks)
    toks = clean_and_tokenize(text)

    # Spellcheck tokens
    toks = spellcheck_token_list(toks)

    # Optional second lemmatize + stoplist
    if lemmatize:
        toks = [_LOOKUP.get(tok, tok) for tok in toks]
    if stoplist:
        toks = [tok for tok in toks if tok not in STOPWORDS]

    # Final normalize and tokenize
    text = " ".join(toks)
    toks = clean_and_tokenize(text)
    return toks


def _build_author_filter(coll, filter_users: List[str] | None, min_docs: int | None) -> Dict[str, Any]:
    """
    Helper to build an author filter for queries based on min_docs and/or filter_users.
    Returns a dict with {"author": {"$in": [...]}} when applicable, otherwise {}.
    """
    if min_docs is not None:
        pipeline: list[dict] = [
            {"$match": {"subreddit": FILTER_SUBREDDIT}},
            {"$group": {"_id": "$author", "post_count": {"$sum": 1}}},
            {"$match": {"post_count": {"$gte": int(min_docs)}}},
        ]
        authors = [doc["_id"] for doc in coll.aggregate(pipeline, allowDiskUse=True)]
        if not authors:
            return {}
        if filter_users:
            authors = [a for a in authors if a in filter_users]
            if not authors:
                return {}
        return {"author": {"$in": authors}}
    elif filter_users:
        return {"author": {"$in": filter_users}}
    else:
        return {}


def _iter_noburp_documents(
    mongo_uri: str = "mongodb://localhost:27017/",
    batch_size: int = 5000,
    filter_users: list[str] | None = None,
    min_docs: int | None = None,
) -> Iterator[str]:
    """Stream raw text docs from reddit.noburp_all where subreddit='noburp'."""
    logging.info("Connecting to MongoDB at %s", mongo_uri)
    coll = pymongo.MongoClient(mongo_uri)[DB_NAME][COLLECTION_NAME]

    # Base query and optional author filter
    query: Dict[str, Any] = {"subreddit": FILTER_SUBREDDIT}
    query.update(_build_author_filter(coll, filter_users, min_docs))

    projection = {"body": 1, "title": 1, "selftext": 1, "_id": 0}
    cursor = coll.find(query, projection=projection, batch_size=batch_size, no_cursor_timeout=True)

    try:
        for doc in cursor:
            body = (doc.get("body") or "").strip()
            if body:
                yield body
                continue  # body takes precedence; match original behavior

            title = (doc.get("title") or "").strip()
            if title:
                yield title

            selftext = (doc.get("selftext") or "").strip()
            if selftext:
                yield selftext
    finally:
        cursor.close()


def _iter_noburp_documents_with_details(
    mongo_uri: str = "mongodb://localhost:27017/",
    batch_size: int = 5000,
    filter_users: list[str] | None = None,
    min_docs: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream dicts containing text + created_utc + author from reddit.noburp_all where subreddit='noburp'.

    For each underlying MongoDB document, this yields one or more entries:
      - If body exists: one entry for the body (and we do NOT also emit title/selftext for that doc,
        mirroring the precedence/continue used in _iter_noburp_documents).
      - Else, we may emit title and selftext entries (if present).
    """
    logging.info("Connecting to MongoDB at %s", mongo_uri)
    coll = pymongo.MongoClient(mongo_uri)[DB_NAME][COLLECTION_NAME]

    query: Dict[str, Any] = {"subreddit": FILTER_SUBREDDIT}
    query.update(_build_author_filter(coll, filter_users, min_docs))

    projection = {
        "body": 1,
        "title": 1,
        "selftext": 1,
        "created_utc": 1,
        "author": 1,
        "_id": 0,
    }
    cursor = coll.find(query, projection=projection, batch_size=batch_size, no_cursor_timeout=True)

    try:
        for doc in cursor:
            created_utc = doc.get("created_utc", None)
            author = doc.get("author", None)

            body = (doc.get("body") or "").strip()
            if body:
                yield {"text": body, "created_utc": created_utc, "author": author}
                continue  # body takes precedence, match original behavior

            title = (doc.get("title") or "").strip()
            if title:
                yield {"text": title, "created_utc": created_utc, "author": author}

            selftext = (doc.get("selftext") or "").strip()
            if selftext:
                yield {"text": selftext, "created_utc": created_utc, "author": author}
    finally:
        cursor.close()


def process_all_noburp(
    mongo_uri: str = "mongodb://localhost:27017/",
    filter_users: list[str] | None = None,
    min_docs: int | None = None,
    max_workers: int | None = None,
    chunksize: int = 2000,
    show_progress: bool = True,
    tokenize: bool = True,
    stoplist: bool = True,
    lemmatize: bool = True,
) -> List[List[str]]:
    """End-to-end: fetch noburp docs → parallel process → collect (returns token lists per emitted text)."""
    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 1)

    doc_iter = _iter_noburp_documents(
        mongo_uri=mongo_uri,
        batch_size=5000,
        filter_users=filter_users,
        min_docs=min_docs,
    )

    results: List[List[str]] = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
    ) as ex:
        worker = partial(_process_one, tokenize=tokenize, stoplist=stoplist, lemmatize=lemmatize)
        mapped = ex.map(worker, doc_iter, chunksize=chunksize)
        if show_progress:
            for toks in tqdm(mapped, unit="doc"):
                results.append(toks)
        else:
            results.extend(mapped)

    return results


def process_all_noburp_details(
    mongo_uri: str = "mongodb://localhost:27017/",
    filter_users: list[str] | None = None,
    min_docs: int | None = None,
    batch_size: int = 5000,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch noburp docs and return a list of dicts:
        {"text": <str>, "created_utc": <int|float|None>, "author": <str|None>}
    One or more entries may be returned per MongoDB document (see iterator behavior).
    """
    entries: List[Dict[str, Any]] = []
    it = _iter_noburp_documents_with_details(
        mongo_uri=mongo_uri,
        batch_size=batch_size,
        filter_users=filter_users,
        min_docs=min_docs,
    )

    if show_progress:
        for item in tqdm(it, unit="doc"):
            entries.append(item)
    else:
        entries.extend(it)

    return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Original processing (tokens)
    cleaned_docs = process_all_noburp(
        max_workers=None,   # use all CPU cores
        chunksize=2000,     # adjust for speed/memory tradeoff
        show_progress=True,
    )
    print(f"Processed {len(cleaned_docs)} docs into tokens.")

    # New details path (raw text + metadata)
    details = process_all_noburp_details(show_progress=True)
    print(f"Collected {len(details)} text+meta entries.")
