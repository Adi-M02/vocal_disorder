"""
evaluate the performance of word2vec expansion terms on manual text
usage:
  python word2vec_expansion/evaluate_expansions_lemmatized.py \
    --expansion_path <path> \
    --manual_dir <dir> \
    [--spellcheck] \
    [--lookup <lemma_lookup.json>] \
    [--ngram <N|<=N>]
"""
import sys
import os
import json
import logging
import argparse
from collections import Counter
from typing import Optional, Tuple, List

# allow importing your project's tokenizer
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize

# Attempt to import spell‐checking tokenizer (if user requested)
try:
    from spellchecker_folder.spellchecker import clean_and_tokenize_spellcheck
except ImportError:
    clean_and_tokenize_spellcheck = None

from query_mongo import return_documents
import datetime


def parse_ngram_filter(arg: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse the --ngram argument into a (min_ngram, max_ngram) pair.
     - "3"   → (3, 3)
     - "<=2" → (1, 2)
    """
    if arg is None:
        return None
    arg = arg.strip()
    if arg.startswith("<="):
        n = int(arg[2:])
        return (1, n)
    else:
        n = int(arg)
        return (n, n)


def load_expansion_terms(
    path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn
) -> List[str]:
    """
    Load and tokenize expansion terms, filtering by ngram_filter if provided.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    terms: List[str] = []
    for value in data.values():
        candidates = value if isinstance(value, list) else [value]
        for term in candidates:
            tokens = tok_fn(term)
            if not tokens:
                continue
            L = len(tokens)
            if ngram_filter is None or (ngram_filter[0] <= L <= ngram_filter[1]):
                terms.append(" ".join(tokens))
    return terms


def load_manual_terms(
    path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn
) -> List[str]:
    """
    Load comma-separated manual terms, tokenize and filter by ngram_filter.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().lower()
    candidates = [t.strip() for t in raw.split(",") if t.strip()]

    cleaned: List[str] = []
    for term in candidates:
        tokens = tok_fn(term)
        if not tokens:
            continue
        L = len(tokens)
        if ngram_filter is None or (ngram_filter[0] <= L <= ngram_filter[1]):
            cleaned.append(" ".join(tokens))
    return cleaned


def load_user_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().lower()
    return [u.strip() for u in raw.split(",") if u.strip()]


def evaluate_terms_performance(
    docs: List[str],
    manual_terms_path: str,
    expansion_terms_path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn
) -> dict:
    def norm(term: str) -> str:
        return " ".join(tok_fn(term))

    manual_terms    = load_manual_terms(manual_terms_path, ngram_filter, tok_fn)
    expansion_terms = load_expansion_terms(expansion_terms_path, ngram_filter, tok_fn)

    manual_norm    = {t: norm(t) for t in manual_terms}
    expansion_norm = {t: norm(t) for t in expansion_terms}
    universe       = set(manual_terms) | set(expansion_terms)

    TP = FP = TN = FN = 0
    fp_terms = set()
    fn_terms = set()

    for doc in docs:
        tokens = tok_fn(doc)
        positions = {}
        for term in universe:
            term_toks = term.split()
            L = len(term_toks)
            for i in range(len(tokens)-L+1):
                if tokens[i:i+L] == term_toks:
                    positions.setdefault(term, []).append((i, i+L))

        used = [False]*len(tokens)
        matched = set()
        for term in sorted(positions, key=lambda t: -len(t.split())):
            for (start, end) in positions[term]:
                if not any(used[start:end]):
                    matched.add(term)
                    for j in range(start, end):
                        used[j] = True
                    break

        manual_found    = matched & set(manual_norm)
        expansion_found = matched & set(expansion_norm)

        for term in universe:
            m = term in manual_found
            e = term in expansion_found
            if m and e:
                TP += 1
            elif not m and e:
                FP += 1
                fp_terms.add(term)
            elif m and not e:
                FN += 1
                fn_terms.add(term)
            else:
                TN += 1

    total     = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    accuracy  = (TP + TN) / total if total else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    base, _ = os.path.splitext(expansion_terms_path)
    ts = datetime.datetime.now().strftime("%m_%d_%H_%M")
    out_path = f"{base}_evaluation_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Expansion JSON: {expansion_terms_path}\n")
        if ngram_filter is None:
            f.write("No n-gram filtering applied\n")
        else:
            f.write(f"Filtered terms to n-grams in [{ngram_filter[0]}, {ngram_filter[1]}]\n")
        f.write(f"=== Evaluation Across {len(docs)} Documents ===\n")
        f.write(f"Total terms evaluated: {total}\n")
        f.write(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall:    {recall:.3f}\n")
        f.write(f"F1 Score:  {f1:.3f}\n")
        f.write(f"Accuracy:  {accuracy:.3f}\n\n")
        if fp_terms:
            f.write(f"Unique False Positives ({len(fp_terms)}):\n")
            f.write("\n".join(sorted(fp_terms)) + "\n\n")
        if fn_terms:
            f.write(f"Unique False Negatives ({len(fn_terms)}):\n")
            f.write("\n".join(sorted(fn_terms)) + "\n")

    print(f"\nResults -- Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
    print(f"Details written to {out_path}")
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expansion JSON against manual terms, with optional n-gram filtering and lemmatization."
    )
    parser.add_argument("--expansion_path", required=True,
        help="Path to the expansion JSON file.")
    parser.add_argument("--manual_dir", required=True,
        help="Directory with manual_terms.txt and users.txt")
    parser.add_argument("--spellcheck", action="store_true",
        help="Use the spell-checking tokenizer.")
    parser.add_argument("--lookup", type=str, default=None,
        help="Path to lemma-lookup JSON; if set, lemmatize tokens via this mapping.")
    parser.add_argument("--ngram", type=str, default=None,
        help="Filter by n-gram length. Use 'N' for exactly N or '<=N' for up to N.")

    args = parser.parse_args()

    if args.spellcheck and clean_and_tokenize_spellcheck is None:
        parser.error("Spell-checking tokenizer requested but not available.")

    # choose base tokenizer
    tok_fn = clean_and_tokenize_spellcheck if args.spellcheck else clean_and_tokenize
    print(f"→ Using {'spell-checking' if args.spellcheck else 'vanilla'} tokenizer")

    # optionally wrap with lemma lookup
    if args.lookup:
        with open(args.lookup, 'r', encoding='utf-8') as f:
            lemma_map = json.load(f)
        base_fn = tok_fn
        def tok_lemmatize(text: str):
            return [lemma_map.get(tok, tok) for tok in base_fn(text)]
        tok_fn = tok_lemmatize
        print(f"→ Applying lemma lookup from {args.lookup}")

    ngram_filter = parse_ngram_filter(args.ngram)

    # verify manual_dir
    manual_terms_path = os.path.join(args.manual_dir, "manual_terms.txt")
    users_path        = os.path.join(args.manual_dir, "users.txt")
    for p in (manual_terms_path, users_path):
        if not os.path.isfile(p):
            parser.error(f"{p} not found.")

    users = load_user_list(users_path)
    if not users:
        parser.error("No users listed in users.txt")

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        filter_users=users
    )
    logging.info("Fetched %d documents for %d users", len(docs), len(users))

    evaluate_terms_performance(
        docs,
        manual_terms_path,
        args.expansion_path,
        ngram_filter,
        tok_fn
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()