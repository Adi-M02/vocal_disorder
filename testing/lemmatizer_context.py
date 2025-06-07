"""
Evaluate Stanza lemmatization on a sampled set of Reddit documents, building a lemma lookup table.

This script:
  1) Fetches all Reddit docs and tokenizes via `clean_and_tokenize_spellcheck`
  2) Samples the minimal set so that each unique token appears ≥3 times
  3) Runs Stanza lemmatizer on each sample doc (no batching) with cuDNN disabled
  4) Builds an in-memory lemma map and flushes to disk once
  5) Constructs a lookup table choosing the most frequent lemma per token
  6) Counts lemma≠form occurrences, extracts contexts, and writes a report

Dependencies:
    pip install stanza tqdm
    python -m stanza.download en

Usage:
    python testing/lemmatizer_context.py \
           --min_count 5 \
           --examples_per_term 3 \
           --output_file testing/lemmatizer_context.txt
"""
import sys
import json
import torch
# Disable cuDNN for RNNs to avoid non-contiguous input errors
torch.backends.cudnn.enabled = False

import logging
import shelve
import atexit
from pathlib import Path
from collections import Counter, defaultdict
import stanza
from tqdm import tqdm
import multiprocessing as mp

# ────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from spellchecker_folder.spellchecker import clean_and_tokenize_spellcheck

# ───────────────────────── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ───────────────────── Lemma cache setup ────────────────────────────────────
CACHE_DIR = SCRIPT_DIR / "lemma_cache_stanza_sample"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_disk = shelve.open(str(CACHE_DIR / "lemma_cache"))
atexit.register(_disk.close)
# Load existing cache
lemma_map = dict(_disk)

# ──────────────────── Initialize Stanza pipeline ────────────────────────────
NLP = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma",
    tokenize_pretokenized=True,   # accept List[List[str]]
    tokenize_no_ssplit=True,      # treat each token-list as one sentence
    use_gpu=False,
    verbose=False,
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

# ────────────────────────── Chunk helper ─────────────────────────────────────
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

# ───────────────────────── Worker setup & function ───────────────────────────
def init_worker():
    # Reinitialize NLP in each process
    global NLP
    NLP = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
        use_gpu=True,
        verbose=False,
    )

def lemmatize_one(toks):
    # toks: List[str]
    doc = NLP([toks])
    sent = doc.sentences[0]
    return [ (w.text, (w.lemma or w.text).lower()) for w in sent.words ]

# ────────────────────────── Main pipeline ───────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample-based Stanza lemmatizer evaluator (multiprocessed)"
    )
    parser.add_argument("--min_count", type=int, default=5,
                        help="only consider tokens whose lemma≠form > this count")
    parser.add_argument("--examples_per_term", type=int, default=3,
                        help="how many example contexts per term")
    parser.add_argument("--output_file", type=Path,
                        default=SCRIPT_DIR / "testing/lemmatizer_context.txt",
                        help="path to write term→contexts output")
    args = parser.parse_args()

    # 1) Fetch & tokenize all docs
    raw_docs = fetch_documents()
    tokenized_all = []
    all_tokens = set()
    logging.info("Tokenizing %d documents…", len(raw_docs))
    for text in tqdm(raw_docs, desc="Tokenizing docs"):
        toks = clean_and_tokenize_spellcheck(text)
        tokenized_all.append(toks)
        all_tokens.update(toks)
    logging.info("Unique tokens: %d", len(all_tokens))

    # 2) Sample docs to cover 3 occurrences per token
    needed = {tok: 3 for tok in all_tokens}
    sample_idxs = []
    for idx, toks in enumerate(tokenized_all):
        if any(needed.get(tok, 0) > 0 for tok in set(toks)):
            sample_idxs.append(idx)
            for tok in toks:
                if needed.get(tok, 0) > 0:
                    needed[tok] -= 1
            if all(v <= 0 for v in needed.values()):
                break
    sample_tok = [tokenized_all[i] for i in sample_idxs]
    logging.info("Selected %d docs for sample", len(sample_tok))

    # 3) Lemmatize sample
    tokenized_docs, lemma_docs = [], []
    logging.info("Lemmatizing %d sampled docs (sequential)…", len(sample_tok))
    for toks in tqdm(sample_tok, desc="Lemmatizing sample"):
        # treat toks as one pre-tokenized sentence
        doc = NLP([toks])
        sent = doc.sentences[0]
        lems = []
        for w in sent.words:
            tok = w.text
            lemma = (w.lemma or tok).lower()
            lems.append(lemma)
            if tok not in lemma_map:
                lemma_map[tok] = lemma
        tokenized_docs.append(toks)
        lemma_docs.append(lems)

    # 4) Persist new lemmas once
    new_cnt = sum(1 for tok in lemma_map if tok not in _disk)
    for tok, lem in lemma_map.items():
        if tok not in _disk:
            _disk[tok] = lem
    logging.info("Cached %d new lemma entries", new_cnt)

    # 5) Build lookup table
    lemma_options = defaultdict(Counter)
    for toks, lems in zip(tokenized_docs, lemma_docs):
        for tok, lem in zip(toks, lems):
            lemma_options[tok][lem] += 1
    lookup_map = {tok: ctr.most_common(1)[0][0] for tok, ctr in lemma_options.items()}
    lookup_path = SCRIPT_DIR / "lemma_lookup.json"
    with lookup_path.open("w", encoding="utf-8") as lf:
        json.dump(lookup_map, lf, ensure_ascii=False, indent=2)
    logging.info("Saved lemma lookup to %s", lookup_path)

    # 6) Count lemma≠form and extract contexts
    counts = Counter()
    for toks, lems in zip(tokenized_docs, lemma_docs):
        for tok, lem in zip(toks, lems):
            if tok != lem:
                counts[tok] += 1
    logging.info("%d tokens lemma≠form", len(counts))

    frequent = {tok: c for tok, c in counts.items() if c > args.min_count}
    logging.info("%d tokens > %d occurrences", len(frequent), args.min_count)
    contexts = defaultdict(list)
    window = 5
    for toks in tokenized_docs:
        for i, tok in enumerate(toks):
            if tok in frequent and len(contexts[tok]) < args.examples_per_term:
                start = max(0, i-window)
                end = min(len(toks), i+window+1)
                contexts[tok].append(" ".join(toks[start:end]))
        if all(len(contexts[t])>=args.examples_per_term for t in frequent):
            break

    # 7) Write report
    out = args.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for tok, c in sorted(frequent.items(), key=lambda x: x[1], reverse=True):
            lemma_final = lookup_map.get(tok, lemma_map[tok])
            f.write(f"term: {tok}\nfrequency: {c}\nlemmatized_to: {lemma_final}\n")
            for i, sn in enumerate(contexts[tok],1):
                f.write(f"context {i}: {sn}\n")
            f.write("\n")
    logging.info("Report written to %s", out)

if __name__ == "__main__":
    main()