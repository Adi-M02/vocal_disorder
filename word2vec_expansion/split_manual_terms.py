"""get a percent of terms to use as seed terms
usage: python word2vec_expansion/split_manual_terms.py --manual_terms <path> --outdir <path> --seed_percent <float>"""
import argparse
import sys
import json
import logging
from pathlib import Path
import random

sys.path.append('../vocal_disorder')
from utils.text_pipeline import process_text

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
    p.add_argument('--min_ngram', '-n', type=int, default=None,
                   help="Minimum number of tokens per term (after normalization)")
    args = p.parse_args()

    if not 0.0 <= args.seed_percent <= 1.0:
        p.error("--seed_percent must be between 0.0 and 1.0")

    # 1) load raw comma-separated terms (phrases)
    text = Path(args.manual_terms).read_text(encoding='utf-8')
    raw_terms = [t.strip() for t in text.split(',') if t.strip()]
    logging.info(f"Loaded {len(raw_terms)} raw terms")

    # 2) normalize each phrase but keep it as one unit (list of tokens per term)
    processed_terms = [process_text(term) for term in raw_terms]

    # 2a) remove stopwords BEFORE n-gram length filtering; log any changes
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stopword_log_path = outdir / f"{args.max_ngram}_gram_stopword_removals.jsonl"

    changed_count = 0
    empty_after_removal = 0
    with stopword_log_path.open('w', encoding='utf-8') as logf:
        terms_no_stop = []
        for raw, toks in zip(raw_terms, processed_terms):
            kept, removed = remove_stopwords(toks)
            if removed:
                changed_count += 1
                record = {
                    "raw_term": raw,
                    "processed_tokens": toks,
                    "removed_stopwords": removed,
                    "after_removal": kept,
                    "note": "empty_after_removal" if len(kept) == 0 else "ok"
                }
                json.dump(record, logf, ensure_ascii=False)
                logf.write("\n")
            if len(kept) == 0:
                empty_after_removal += 1
            terms_no_stop.append(kept)

    logging.info(f"Stopword changes: {changed_count} terms affected; "
                 f"{empty_after_removal} became empty after removal")
    logging.info(f"Wrote stopword-change log to {stopword_log_path}")

    # Replace processed_terms with stopword-removed version
    processed_terms = terms_no_stop

    # 2b) filter by n-gram length AFTER stopword removal
    #     (drop empties automatically by the length checks)
    filtered_terms = [
        toks for toks in processed_terms
        if len(toks) <= args.max_ngram
        and (args.min_ngram is None or len(toks) > args.min_ngram)
    ]

    # remove duplicates (list-of-tokens -> tuple -> set -> list)
    filtered_terms = [list(t) for t in {tuple(t) for t in filtered_terms}]

    # 3) shuffle & split by seed_percent *of terms*, not tokens
    random.shuffle(filtered_terms)
    n_seed = int(len(filtered_terms) * args.seed_percent)
    seed_terms = filtered_terms[:n_seed]

    # 5) convert each token list into a space-joined phrase
    seed_strings = [" ".join(tokens) for tokens in seed_terms]

    # 6) write seed_terms.json
    seed_path = outdir / f"{args.max_ngram}_gram_seed_terms.json"
    with seed_path.open('w', encoding='utf-8') as f:
        json.dump({'seed_terms': seed_strings}, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote {len(seed_strings)} seed terms to {seed_path}")
    logging.info(f"Total kept after filtering: {len(filtered_terms)} "
                 f"(from {len(raw_terms)} raw terms)")

if __name__ == "__main__":
    main()
