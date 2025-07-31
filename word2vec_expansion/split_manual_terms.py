"""get a percent of terms to use as seed terms
usage: python word2vec_expansion/split_manual_terms.py --manual_terms <path> --outdir <path> --seed_percent <float>"""
import argparse
import sys
import json
from pathlib import Path
import random
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list

def load_lookup(path: str) -> dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

lookup = load_lookup("testing/lemma_lookup.json")

def normalize_term(term: str) -> str:
    """
    Split term into tokens, apply lookup+spellcheck, then re-join
    so we preserve multi-word phrases as single items.
    """
    toks = clean_and_tokenize(term)
    toks = [lookup.get(t, t) for t in toks]
    toks = spellcheck_token_list(toks)
    toks = [lookup.get(t, t) for t in toks]
    return " ".join(toks)


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
    args = p.parse_args()

    if not 0.0 <= args.seed_percent <= 1.0:
        p.error("--seed_percent must be between 0.0 and 1.0")

    # 1) load raw comma-separated terms (phrases)
    text = Path(args.manual_terms).read_text(encoding='utf-8')
    raw_terms = [t.strip() for t in text.split(',') if t.strip()]

    # 2) normalize each phrase but keep it as one unit
    normalized_terms = [normalize_term(term) for term in raw_terms]

    # 3) shuffle & split by seed_percent *of terms*, not tokens
    random.shuffle(normalized_terms)
    n_seed = int(len(normalized_terms) * args.seed_percent)
    seed_terms = normalized_terms[:n_seed]

    # 4) ensure outdir exists
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 5) write seed_terms.json
    seed_path = outdir / "seed_terms.json"
    with open(seed_path, 'w', encoding='utf-8') as f:
        json.dump({'seed_terms': seed_terms},
                  f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(seed_terms)} seed terms to {seed_path}")


if __name__ == "__main__":
    main()