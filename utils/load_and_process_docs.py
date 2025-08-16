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
LOOKUP_PATH = "testing/combined_lemmas.json"

# -------- Worker globals --------
_LOOKUP: Dict[str, str] | None = None

def _init_worker():
    """Runs once per worker process to load heavy resources."""
    global _LOOKUP
    _LOOKUP = load_lookup(LOOKUP_PATH)

    # Avoid oversubscription for BLAS threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _process_one(text: str, stoplist = True, lemmatize = True) -> List[str]:
    """Your process_text logic using preloaded lookup."""
    assert _LOOKUP is not None, "Worker not initialized; missing lookup."

    toks = clean_and_tokenize(text)
    if lemmatize:
        toks = [_LOOKUP.get(tok, tok) for tok in toks]

    text = " ".join(toks)
    toks = clean_and_tokenize(text)

    toks = spellcheck_token_list(toks)

    if lemmatize:
        toks = [_LOOKUP.get(tok, tok) for tok in toks]
    if stoplist:
        toks = [tok for tok in toks if tok not in STOPWORDS]

    text = " ".join(toks)
    toks = clean_and_tokenize(text)
    return toks

def _iter_noburp_documents(
    mongo_uri: str = "mongodb://localhost:27017/",
    batch_size: int = 5000,
    filter_users: list[str] | None = None,
    min_docs: int | None = None,
) -> Iterator[str]:
    """Stream raw text docs from reddit.noburp_all where subreddit='noburp'."""
    logging.info("Connecting to MongoDB at %s", mongo_uri)
    coll = pymongo.MongoClient(mongo_uri)[DB_NAME][COLLECTION_NAME]

    query: Dict[str, Any] = {"subreddit": FILTER_SUBREDDIT}

    if min_docs is not None:
        pipeline: list[dict] = [
            {"$match": {"subreddit": FILTER_SUBREDDIT}},
            {"$group": {"_id": "$author", "post_count": {"$sum": 1}}},
            {"$match": {"post_count": {"$gte": int(min_docs)}}},
        ]
        authors = [doc["_id"] for doc in coll.aggregate(pipeline, allowDiskUse=True)]
        if not authors:
            return
        if filter_users:
            authors = [a for a in authors if a in filter_users]
            if not authors:
                return
        query["author"] = {"$in": authors}
    elif filter_users:
        query["author"] = {"$in": filter_users}

    projection = {"body": 1, "title": 1, "selftext": 1, "_id": 0}
    cursor = coll.find(query, projection=projection, batch_size=batch_size, no_cursor_timeout=True)

    try:
        for doc in cursor:
            body = (doc.get("body") or "").strip()
            if body:
                yield body
                continue
            title = (doc.get("title") or "").strip()
            if title:
                yield title
            selftext = (doc.get("selftext") or "").strip()
            if selftext:
                yield selftext
    finally:
        cursor.close()

def process_all_noburp(
    mongo_uri: str = "mongodb://localhost:27017/",
    filter_users: list[str] | None = None,
    min_docs: int | None = None,
    max_workers: int | None = None,
    chunksize: int = 2000,
    show_progress: bool = True,
    stoplist: bool = True,
    lemmatize: bool = True
) -> List[List[str]]:
    """End-to-end: fetch noburp docs → parallel process → collect."""
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
        worker = partial(_process_one, stoplist=stoplist, lemmatize=lemmatize)
        mapped = ex.map(worker, doc_iter, chunksize=chunksize)
        if show_progress:
            for toks in tqdm(mapped, unit="doc"):
                results.append(toks)
        else:
            results.extend(mapped)

    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaned_docs = process_all_noburp(
        max_workers=None,   # use all CPU cores
        chunksize=2000,      # adjust for speed/memory tradeoff
        show_progress=True,
    )
    print(f"Processed {len(cleaned_docs)} docs.")
