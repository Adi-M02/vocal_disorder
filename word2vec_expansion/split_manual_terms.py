"""get a percent of terms to use as seed terms
usage: python word2vec_expansion/split_manual_terms.py --manual_terms <path> --outdir <path> --seed_percent <float>"""
import argparse
import sys
import json
from pathlib import Path
import random
sys.path.append('../vocal_disorder')
from utils.text_pipeline import process_text


def main():
    p = argparse.ArgumentParser(
        description="Split manual terms into a JSON seed set and a comma-separated eval file"
    )
    p.add_argument('--manual_terms', '-i', required=True,
                   help="Path to input TXT file (comma-separated terms)")
    p.add_argument('--outdir', '-o', required=True,
                   help="Directory to write `seed_terms.json` and `eval_terms.txt`")
    p.add_argument('--seed_percent', '-s', type=float, required=True,
                   help="Fraction (0.0–1.0) of terms to include in the seed JSON")
    p.add_argument('--max_ngram', '-m', type=int, required=True,
                   help="Maximum number of tokens per term (after normalization)")
    args = p.parse_args()

    if not 0.0 <= args.seed_percent <= 1.0:
        p.error("--seed_percent must be between 0.0 and 1.0")

    # 1) load raw comma-separated terms (phrases)
    text = Path(args.manual_terms).read_text(encoding='utf-8')
    raw_terms = [t.strip() for t in text.split(',') if t.strip()]

    # 2) normalize each phrase but keep it as one unit
    processed_terms = [process_text(term) for term in raw_terms]

    # 2b) filter out anything longer than max_ngram tokens
    processed_terms = [
        toks for toks in processed_terms
        if len(toks) <= args.max_ngram
    ]
    # 3) shuffle & split by seed_percent *of terms*, not tokens
    random.shuffle(processed_terms)
    n_seed = int(len(processed_terms) * args.seed_percent)
    seed_terms = processed_terms[:n_seed]

    # 4) ensure outdir exists
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 5) convert each token list into a space-joined phrase
    seed_strings = [" ".join(tokens) for tokens in seed_terms]

    # 6) write seed_terms.json
    seed_path = outdir / f"{args.max_ngram}_gram_seed_terms.json"
    with open(seed_path, 'w', encoding='utf-8') as f:
        json.dump({'seed_terms': seed_strings},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()