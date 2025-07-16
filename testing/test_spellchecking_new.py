#!/usr/bin/env python3
import json
import sys
import argparse
import logging
import shelve
import atexit
import multiprocessing as mp
from pathlib import Path

from spellchecker import SpellChecker

# make sure this points at your project’s tokenizer
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize

# ───────────────────── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def _init_worker(domain_tokens: list[str]):
    global _SPELL
    _SPELL = SpellChecker()
    _SPELL.word_frequency.load_words(domain_tokens)
    for tok in domain_tokens:
        if any(ch.isdigit() for ch in tok):
            _SPELL.word_frequency.add(tok)

def _worker(tok: str) -> tuple[str, str]:
    if any(ch.isdigit() for ch in tok):
        return tok, tok
    corr = _SPELL.correction(tok) or tok
    return tok, corr

def load_lookup(path: str) -> dict[str, str]:
    """Load your lemma_lookup.json into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Clean, lemmatize, and spell-check a comma-separated list of manual terms."
    )
    parser.add_argument("--manual_terms",  required=True,
                        help="Path to manual_terms.txt (comma-separated)")
    parser.add_argument("--lookup_map",    required=True,
                        help="Path to lemma_lookup.json")
    parser.add_argument("--output", default="processed_manual_terms.tsv",
                        help="Where to write the TSV of original vs. processed terms")
    args = parser.parse_args()

    # ─────────── 1) Load raw manual terms ────────────────────
    with open(args.manual_terms, encoding="utf-8") as f:
        raw_terms = [t.strip()
                     for line in f
                     for t in line.split(",")
                     if t.strip()]
    logging.info("Loaded %d raw terms", len(raw_terms))

    # ─────────── 2) Load your lemma lookup map ───────────────
    lookup_map = load_lookup(args.lookup_map)
    logging.info("Loaded %d lemma mappings", len(lookup_map))

    # ─────────── 3) Build domain-specific tokens ─────────────
    #    (we apply lemmatization here, not just plain tokens)
    domain_tokens: set[str] = set()
    for term in raw_terms:
        toks    = clean_and_tokenize(term)
        lemtoks = [ lookup_map.get(tok, tok) for tok in toks ]
        domain_tokens.update(lemtoks)
    logging.info("Built %d domain tokens (lemmatized)", len(domain_tokens))

    # ─────────── 4) Initialize on-disk cache ─────────────────
    CACHE_DIR = Path(__file__).parent / "spell_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = str(CACHE_DIR / "manual_terms_cache")
    _disk_cache = shelve.open(cache_path)
    atexit.register(_disk_cache.close)
    logging.info("Using cache at %s", cache_path)

    # ─────────── 5) Spell-check unique tokens in parallel ────
    unique_tokens = list(domain_tokens)
    todo = [tok for tok in unique_tokens if tok not in _disk_cache]
    if todo:
        n_proc = min(mp.cpu_count() - 1, len(todo))
        logging.info("Spell-checking %d tokens on %d procs…", len(todo), n_proc)
        with mp.Pool(
            processes=n_proc,
            initializer=_init_worker,
            initargs=(unique_tokens,),
        ) as pool:
            for orig, corr in pool.imap_unordered(_worker, todo, chunksize=100):
                _disk_cache[orig] = corr
    else:
        logging.info("All tokens already cached – skipping spell-check.")

    # ─────────── 6) Process each term ────────────────────────
    processed = []
    for term in raw_terms:
        toks    = clean_and_tokenize(term)
        # ← apply lemma lookup again here, just like in your docs example
        lemtoks = [ lookup_map.get(tok, tok) for tok in toks ]
        # then pull the (possibly corrected) version from cache
        corrected = [ _disk_cache.get(tok, tok) for tok in lemtoks ]
        processed.append(" ".join(corrected))

    # ─────────── 7) Write TSV ────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write("original\tprocessed\n")
        for orig, proc in zip(raw_terms, processed):
            # only write if something actually changed
            if orig.strip() != proc.strip():
                fout.write(f"{orig}\t{proc}\n")

    logging.info("Wrote %d changed lines to %s", 
                 sum(1 for o,p in zip(raw_terms,processed) if o.strip()!=p.strip()), 
                 out_path)

    logging.info("Wrote %d lines (plus header) to %s", len(processed), out_path)

if __name__ == "__main__":
    main()
