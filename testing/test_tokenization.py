#!/usr/bin/env python
"""
check_tokenization.py

Load Reddit documents, apply the updated tokenization pipeline (preserving hyphens,
apostrophes, and slashes), then print before/after tokens for 50 random samples.
"""

import sys
import logging
import re
import random
from pathlib import Path
from tqdm import tqdm
import unicodedata

# Adjust this path if needed to point at your vocal_disorder package
sys.path.append("../vocal_disorder")
from query_mongo import return_documents

def load_slang_map(path: str) -> dict[str, str]:
    """
    Reads lines like 'AFAIK=As Far As I Know' and returns a map
    abbr.lower() -> phrase.lower().
    """
    slang = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        abbr, phrase = line.split("=", 1)
        slang[abbr.strip().lower()] = phrase.strip().lower()
    return slang

def expand_slang(text: str, slang_map: dict[str, str]) -> str:
    """
    Replace standalone abbreviations with their full phrase.
    """
    if not slang_map:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in slang_map.keys()) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda m: slang_map[m.group(0).lower()], text)

# ────────────────────────── Logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

# ──────────────────────── 1. Load documents ──────────────────────
docs = list(
    return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        min_docs=None,
        mongo_uri="mongodb://localhost:27017/",
    )
)
logging.info("Fetched %d documents", len(docs))

slang_map = load_slang_map("testing/slang.txt")
if slang_map:
    docs = [expand_slang(text, slang_map) for text in docs]
    logging.info("Applied slang expansion for %d abbreviations", len(slang_map))

# ──────────────────── 2. Tokenization pipeline ──────────────────
# allow internal apostrophes and hyphens, preserve subreddit/user slash
tokeniser   = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'/\-]*[A-Za-z0-9])?", re.I)
url_pattern = re.compile(r'(?:https?://|www\.|[A-Za-z0-9\-]+\.(?:com|org|io|be)/)\S+', re.I)
# remove everything except letters, numbers, apostrophe, hyphen, slash, or space
junk_pattern = re.compile(r"[^A-Za-z0-9'/\- ]+")


def clean_and_tokenize(text: str) -> list[str]:
    """
    Apply URL removal, slash preservation for r/ and u/, junk stripping, then extract tokens.
    """
    clean = unicodedata.normalize("NFKC", text)
    clean = clean.replace("–", "-").replace("—", "-")
    clean = clean.replace("‘", "'").replace("’", "'")
    clean = clean.replace("…", "...")
    clean = clean.replace("\u200B", "").replace("\u00A0", " ")
    # 1) strip URLs
    clean = url_pattern.sub(" ", clean)
    # 2) preserve subreddit/user slashes (r/xxx or u/xxx)
    clean = re.sub(r"\b([ru])/", r"\1<SLASH>", clean, flags=re.I)
    # 3) replace any other slash with space
    clean = clean.replace("/", " ")
    # 4) restore preserved slash
    clean = clean.replace("<SLASH>", "/")
    clean = clean.replace("-", " ")
    # 5) drop other junk but keep letters, apostrophes, hyphens, and spaces
    clean = junk_pattern.sub(" ", clean)
    # 6) tokenize on lowercase
    return tokeniser.findall(clean.lower())

# ──────────────────── 3. Sample & display ───────────────────────
random.seed(42)
samples = random.sample(docs, min(50, len(docs)))

for idx, doc in enumerate(samples, start=1):
    tokens = clean_and_tokenize(doc)
    print(f"\n--- Sample #{idx} ---")
    print("Original:", doc)
    print("Tokens:  ", tokens)
