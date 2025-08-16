"""get a percent of terms to use as seed terms
usage: python word2vec_expansion/split_manual_terms.py --manual_terms <path> --outdir <path> --seed_percent <float>"""
import argparse
import sys
import json
import logging
from pathlib import Path
import random

sys.path.append('../vocal_disorder')
from utils.text_pipeline import process_text, remove_unigram_stopwords
# custom stopword list
from utils.stopwords import STOPWORDS
STOPWORDS = set(STOPWORDS)


def remove_stopwords(tokens):
    """Return (filtered_tokens, removed_tokens) given a list of tokens."""
    removed = [t for t in tokens if t.lower() in STOPWORDS]
    kept = [t for t in tokens if t.lower() not in STOPWORDS]
    return kept, removed


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Split manual terms into a JSON seed set and a comma-separated eval file"
    )
    p.add_argument('--manual_terms', '-i', required=True,
                   help="Path to input TXT file (comma-separated terms)")
    p.add_argument('--outdir', '-o', required=True,
                   help="Directory to write `seed_terms.json`")
    p.add_argument('--seed_percent', '-s', type=float, required=True,
                   help="Fraction (0.0–1.0) of terms to include in the seed JSON")
    p.add_argument('--max_ngram', '-m', type=int, required=True,
                   help="Maximum number of tokens per term (after normalization)")
    p.add_argument('--min_ngram', '-n', type=int, default=1,
                   help="Minimum number of tokens per term (after normalization)")
    args = p.parse_args()

    if not 0.0 <= args.seed_percent <= 1.0:
        p.error("--seed_percent must be between 0.0 and 1.0")

    # 1) load raw comma-separated terms (phrases)
    text = Path(args.manual_terms).read_text(encoding='utf-8')
    raw_terms = [t.strip() for t in text.split(',') if t.strip()]
    logging.info(f"Loaded {len(raw_terms)} raw terms")

    # 2) normalize each phrase but keep it as one unit (list of tokens per term)
    processed_terms = [process_text(term, stoplist=False) for term in raw_terms]

    outdir = Path(args.outdir)

    # 2b) filter by n-gram length AFTER stopword removal
    #     (drop empties automatically by the length checks)
    filtered_terms = [
        toks for toks in processed_terms
        if len(toks) <= args.max_ngram
        and len(toks) >= args.min_ngram
    ]

    # remove duplicates (list-of-tokens -> tuple -> set -> list)
    filtered_terms = [list(t) for t in {tuple(t) for t in filtered_terms}]

    # 3) shuffle & split by seed_percent *of terms*, not tokens
    random.shuffle(filtered_terms)
    n_seed = int(len(filtered_terms) * args.seed_percent)
    seed_terms = filtered_terms[:n_seed]

    # 5) convert each token list into a _-joined phrase
    seed_strings = ["_".join(tokens) for tokens in seed_terms]

    # 6) write seed_terms.json
    if args.min_ngram != args.max_ngram:
        seed_filename = f"{args.min_ngram}_to_{args.max_ngram}_gram_seed_terms.json"
    else:
        seed_filename = f"{args.max_ngram}_gram_seed_terms.json"
    seed_path = outdir / seed_filename
    with seed_path.open('w', encoding='utf-8') as f:
        json.dump({'seed_terms': seed_strings}, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote {len(seed_strings)} seed terms to {seed_path}")
    logging.info(f"Total kept after filtering: {len(filtered_terms)} "
                 f"(from {len(raw_terms)} raw terms)")

if __name__ == "__main__":
    main()
