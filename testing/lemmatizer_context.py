#!/usr/bin/env python
"""
Evaluate spaCy lemmatization on Reddit documents.

For every token whose lemma ≠ surface form, count how often that happens
across the corpus, then (for high-frequency cases) write example contexts.

Dependencies
────────────
pip install spacy tqdm
python -m spacy download en_core_web_sm   # or _lg if you like

Usage
─────
python testing/lemmatizer_context.py \
       --min_count 5 \
       --examples_per_term 3 \
       --output_file testing/lemmatizer_context.txt
"""
import json
import sys
import logging
import shelve
import atexit
from pathlib import Path
from collections import Counter, defaultdict

import stanza
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Project-specific imports (adjust path if needed)
SCRIPT_DIR = Path(__file__).parent
sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# ───────────────────────── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ─────────────────────  Lemma cache setup ─────────────────────────────────
CACHE_DIR = SCRIPT_DIR / "lemma_cache_stanza_doc"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_disk = shelve.open(str(CACHE_DIR / "lemma_cache"))
atexit.register(_disk.close)

# ─────────────────── Initialize stanza pipeline once ───────────────────────
# We use `tokenize_pretokenized=True` so that stanza accepts our pre-split tokens.
NLP = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma",
    tokenize_pretokenized=True,
    tokenize_no_ssplit=True,
    verbose=False,
    use_gpu=True,
)

# ─────────────────────── Fetch documents ────────────────────────────────────
def fetch_documents():
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    logging.info("Fetched %d documents", len(docs))
    return docs

# ────────────────────────── Main pipeline ───────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate stanza lemmatizer; write contexts for high-frequency changes."
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=100,
        help="Only consider tokens whose lemma ≠ form more than this many times."
    )
    parser.add_argument(
        "--examples_per_term",
        type=int,
        default=3,
        help="How many example contexts to extract per frequent token."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to write the term→contexts TXT output."
    )
    args = parser.parse_args()

    # 1) Fetch and tokenize all documents
    raw_docs = fetch_documents()
    tokenized_docs = []
    all_tokens = set()

    logging.info("Tokenizing %d documents…", len(raw_docs))
    for text in tqdm(raw_docs, desc="Tokenizing"):
        toks = clean_and_tokenize(text)
        tokenized_docs.append(toks)
        all_tokens.update(toks)
    logging.info("Unique tokens extracted: %d", len(all_tokens))

    # 2) Build or extend lemma cache
    missing = [tok for tok in all_tokens if tok not in _disk]
    if missing:
        logging.info("Lemmatizing %d tokens via document-level stanza…", len(missing))
        for doc_tokens in tqdm(tokenized_docs, desc="Stanza lemma pass"):
            if not missing:
                break  # cache is already full
            # Pass the token list as a single "sentence"
            doc = NLP([doc_tokens])
            # stanza produces exactly one sentence when tokenize_no_ssplit=True
            for w in doc.sentences[0].words:
                tok = w.text
                if tok in _disk:
                    continue
                lemma = (w.lemma or tok).lower()
                _disk[tok] = lemma
                if tok in missing:
                    missing.remove(tok)
        logging.info("Cache populated; %d tokens still missing.", len(missing))
    else:
        logging.info("All tokens already cached – skipping lemmatization.")

    # 3) Count how often each token changes after lemmatization
    counts = Counter()
    for toks in tqdm(tokenized_docs, desc="Counting changes"):
        for tok in toks:
            lem = _disk.get(tok, tok)
            if lem != tok:
                counts[tok] += 1
    logging.info("Found %d distinct tokens with lemma ≠ form", len(counts))

    # 4) Apply frequency threshold
    frequent = {tok: cnt for tok, cnt in counts.items() if cnt > args.min_count}
    logging.info("%d tokens lemmatized > %d times", len(frequent), args.min_count)
    if not frequent:
        logging.info("No tokens exceed the threshold; exiting.")
        return

    # 5) Extract example contexts (±5 tokens around each occurrence)
    window_radius = 5
    contexts_per_term = defaultdict(list)
    logging.info("Collecting up to %d contexts per token…", args.examples_per_term)
    for toks in tqdm(tokenized_docs, desc="Collecting contexts"):
        for i, tok in enumerate(toks):
            if tok in frequent and len(contexts_per_term[tok]) < args.examples_per_term:
                start = max(0, i - window_radius)
                end = min(len(toks), i + window_radius + 1)
                snippet = " ".join(toks[start:end])
                contexts_per_term[tok].append(snippet)
        if all(len(contexts_per_term[tok]) >= args.examples_per_term for tok in frequent):
            break

    # 6) Write output in descending-frequency order
    out_path = args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_tokens = sorted(frequent.items(), key=lambda x: x[1], reverse=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for token, cnt in sorted_tokens:
            fout.write(f"term: {token}\n")
            fout.write(f"frequency: {cnt}\n")
            fout.write(f"lemmatized_to: {_disk.get(token, token)}\n")
            for idx, snippet in enumerate(contexts_per_term.get(token, []), start=1):
                fout.write(f"context {idx}: {snippet}\n")
            fout.write("\n")

    logging.info(
        "Wrote contexts for %d tokens (threshold > %d) to %s",
        len(sorted_tokens),
        args.min_count,
        out_path
    )

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()