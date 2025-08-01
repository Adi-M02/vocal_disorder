import json
import sys
import argparse
import logging
from pathlib import Path

# make sure this points at your project’s tokenizer
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list

# ───────────────────── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from utils.load_lemmatizer import load_lookup

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique tokens, spell-check + lemmatize them, and emit only the changed ones."
    )
    parser.add_argument("--manual_terms", required=True,
                        help="Path to manual_terms.txt (comma-separated)")
    parser.add_argument("--lookup_map", required=True,
                        help="Path to lemma_lookup.json")
    parser.add_argument("--output", default="changed_tokens.tsv",
                        help="Where to write the TSV of original vs. processed tokens")
    args = parser.parse_args()

    # 1) Load raw manual terms (comma-separated phrases)
    with open(args.manual_terms, encoding="utf-8") as f:
        raw_terms = [t.strip()
                     for line in f
                     for t in line.split(",")
                     if t.strip()]
    logging.info("Loaded %d raw phrases", len(raw_terms))

    # 2) Load your lemma lookup map
    lookup_map = load_lookup(args.lookup_map)
    logging.info("Loaded %d lemma mappings", len(lookup_map))

    # 3) Gather all unique tokens
    unique_tokens: set[str] = set()
    for term in raw_terms:
        toks = clean_and_tokenize(term)
        unique_tokens.update(toks)
    logging.info("Extracted %d unique tokens", len(unique_tokens))

    # 4) Process each token: lemmatize → spellcheck → lemmatize
    processed_map: dict[str, str] = {}
    for tok in unique_tokens:
        # first lemmatize
        lemtok = lookup_map.get(tok, tok)
        # then spell-check
        corr   = spellcheck_token_list([lemtok])[0]
        # then lemmatize again
        final  = lookup_map.get(corr, corr)
        processed_map[tok] = final

    # 5) Write out only those tokens that actually changed (case-insensitive)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write("original\tprocessed\n")
        for orig in sorted(unique_tokens):
            proc = processed_map[orig]
            if orig.strip().lower() != proc.strip().lower():
                fout.write(f"{orig}\t{proc}\n")

    changed = sum(1 for o,p in processed_map.items()
                  if o.strip().lower() != p.strip().lower())
    logging.info("Wrote %d changed tokens to %s", changed, out_path)

if __name__ == "__main__":
    main()
