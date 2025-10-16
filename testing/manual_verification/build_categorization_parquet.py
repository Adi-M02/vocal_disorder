#!/usr/bin/env python3
"""
Build a Parquet of sampled posts and comments from the noburp collection.

Rules
- Post condition: document has selftext with at least 88 words. Count words by splitting on spaces
  after collapsing whitespace. Write "<title>\\n\\n<selftext>" into column "post".
- Comment condition: document has body with at least 31 words (same counting rule). Write into column "comment".
- Always include author and created_utc.
- Random sample of 500 rows for each condition (reservoir sampling). If fewer qualify, writes what is available.

Output schema
  author (as-is)
  created_utc (as-is)
  post (string or None)
  comment (string or None)
"""

import sys
import re
import random
import argparse
from typing import Dict, Any, List

import pandas as pd

# Import the metadata-preserving iterator you added
# Adjust path if needed for your repo layout
sys.path.append("../vocal_disorder")
from utils.load_and_process_docs import stream_noburp_with_meta  # noqa: E402


_space_re = re.compile(r"\s+")


def normalize_ws(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _space_re.sub(" ", s.strip())


def word_count_spaces(s: str) -> int:
    s2 = normalize_ws(s)
    if not s2:
        return 0
    # Count by splitting on single spaces
    return len(s2.split(" "))


def reservoir_add(reservoir: List[Dict[str, Any]], k: int, item: Dict[str, Any], seen: int) -> int:
    """
    Reservoir sampling update for one stream.
    'seen' is the number of qualifying items observed before this item.
    """
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        r = random.randint(0, seen)  # inclusive
        if r < k:
            reservoir[r] = item
    return seen + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    ap.add_argument("--post-min-words", type=int, default=88)
    ap.add_argument("--comment-min-words", type=int, default=31)
    ap.add_argument("--post-sample", type=int, default=500)
    ap.add_argument("--comment-sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--out", required=True, help="Output Parquet path, for example noburp_sample.parquet")
    ap.add_argument("--compression", default="zstd",
                    choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    args = ap.parse_args()

    random.seed(args.seed)

    posts_res: List[Dict[str, Any]] = []
    comments_res: List[Dict[str, Any]] = []
    seen_posts = 0
    seen_comments = 0

    # Stream documents with metadata from the noburp collection
    for doc in stream_noburp_with_meta(mongo_uri=args.mongo_uri):
        # Post condition: has selftext with enough words
        st_raw = doc.get("selftext") or ""
        if st_raw.strip():
            if word_count_spaces(st_raw) >= args.post_min_words:
                title = normalize_ws(doc.get("title") or "")
                st = normalize_ws(st_raw)
                post_text = f"{title}\n\n{st}" if title else st
                record_p = {
                    "author": doc.get("author"),
                    "created_utc": doc.get("created_utc"),
                    "post": post_text,
                    "comment": None,
                }
                seen_posts = reservoir_add(posts_res, args.post_sample, record_p, seen_posts)

        # Comment condition: has body with enough words
        body_raw = doc.get("body") or ""
        if body_raw.strip():
            if word_count_spaces(body_raw) >= args.comment_min_words:
                record_c = {
                    "author": doc.get("author"),
                    "created_utc": doc.get("created_utc"),
                    "post": None,
                    "comment": normalize_ws(body_raw),
                }
                seen_comments = reservoir_add(comments_res, args.comment_sample, record_c, seen_comments)

    # Assemble DataFrame with required columns
    cols = ["author", "created_utc", "post", "comment"]
    df_posts = pd.DataFrame(posts_res, columns=cols)
    df_comments = pd.DataFrame(comments_res, columns=cols)
    df = pd.concat([df_posts, df_comments], ignore_index=True)

    compression = None if args.compression == "none" else args.compression
    df.to_parquet(args.out, engine="pyarrow", compression=compression)

    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Posts sampled: {len(df_posts)} of {seen_posts} qualifying seen")
    print(f"Comments sampled: {len(df_comments)} of {seen_comments} qualifying seen")


if __name__ == "__main__":
    main()
