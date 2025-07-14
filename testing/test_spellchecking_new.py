#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from spellchecker import SpellChecker

# ensure you’ve installed the right package:
#   pip uninstall spellchecker
#   pip install pyspellchecker

# make sure this points at your project’s tokenizer
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Clean, lemmatize, and spell-check a comma-separated list of manual terms."
    )
    parser.add_argument(
        "--manual_terms",
        required=True,
        help="Path to manual_terms.txt (comma-separated)"
    )
    parser.add_argument(
        "--lookup_map",
        required=True,
        help="Path to lemma_lookup.json"
    )
    parser.add_argument(
        "--output",
        default="processed_manual_terms.tsv",
        help="Where to write the TSV of original vs. processed terms"
    )
    args = parser.parse_args()

    # 1) Load raw manual terms
    with open(args.manual_terms, encoding="utf-8") as f:
        raw_terms = [
            t.strip()
            for line in f
            for t in line.split(",")
            if t.strip()
        ]

    # 2) Load your lemma lookup map
    with open(args.lookup_map, encoding="utf-8") as f:
        lookup_map: dict = json.load(f)

    # 3) Prepare tokenizer and spell-checker
    tok_fn = clean_and_tokenize

    # Build a set of “domain” tokens so SpellChecker won’t try to change them
    domain_tokens = set()
    for term in raw_terms:
        toks   = tok_fn(term)
        lemmas = [lookup_map.get(tok, tok) for tok in toks]
        domain_tokens.update(lemmas)

    spell = SpellChecker()
    spell.word_frequency.load_words(domain_tokens)

    # 4) Process each term
    processed = []
    for term in raw_terms:
        toks      = tok_fn(term)
        lemmas    = [lookup_map.get(tok, tok) for tok in toks]
        corrected = [spell.correction(tok) or tok for tok in lemmas]
        processed.append(" ".join(corrected))

    # 5) Write TSV: original \t processed
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write("original\tprocessed\n")
        for orig, proc in zip(raw_terms, processed):
            fout.write(f"{orig}\t{proc}\n")

    print(f"Wrote {len(processed)} lines (plus header) to {out_path}")

if __name__ == "__main__":
    main()
