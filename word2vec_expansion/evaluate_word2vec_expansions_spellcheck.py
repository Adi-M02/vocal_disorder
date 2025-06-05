"""
evaluate the performance of word2vec expansion terms on manual text
usage: python word2vec_expansion/evaluate_word2vec_expansions_spellcheck.py --expansion_path <path> --manual_dir <dir> [--spellcheck] [--ngram <1|2|3>]"""
import sys
import os
import json
import logging
import argparse
from collections import Counter
from typing import Optional

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


def load_expansion_terms(path: str, ngram: Optional[int]) -> list[str]:
    """
    Load expansion terms from a JSON file, flatten values into a list,
    tokenize each term with TOK_FN, optionally filter by token length == ngram,
    and return a list of cleaned strings.
    If ngram is None, accept all lengths.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    terms: list[str] = []
    for value in data.values():
        if isinstance(value, list):
            for term in value:
                tokens = TOK_FN(term)
                if tokens and (ngram is None or len(tokens) == ngram):
                    terms.append(' '.join(tokens))
        else:
            tokens = TOK_FN(value)
            if tokens and (ngram is None or len(tokens) == ngram):
                terms.append(' '.join(tokens))
    return terms


def load_manual_terms(path: str, ngram: Optional[int]) -> list[str]:
    """
    Load comma-separated manual terms from a TXT file, tokenize each term with TOK_FN,
    optionally filter by token length == ngram, and return a list of cleaned strings.
    If ngram is None, accept all lengths.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    raw_terms = [t.strip() for t in content.split(',') if t.strip()]

    cleaned_terms: list[str] = []
    for term in raw_terms:
        tokens = TOK_FN(term)
        if tokens and (ngram is None or len(tokens) == ngram):
            cleaned_terms.append(' '.join(tokens))
    return cleaned_terms


def load_user_list(path: str) -> list[str]:
    """
    Load comma-separated usernames from users.txt and return a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    users = [u.strip() for u in content.split(',') if u.strip()]
    return users


def evaluate_terms_performance(
    docs: list[str],
    manual_terms_path: str,
    expansion_terms_path: str,
    ngram: Optional[int]
) -> None:
    """
    For each document, tokenize and detect manual vs expansion terms.
    Computes TP, FP, TN, FN across each term in the union of manual and expansion sets,
    then aggregates totals, calculates precision, recall, accuracy, and logs false positives
    and false negatives. Writes a summary to "<expansion_terms_path>_evaluation.txt".
    """
    # Normalization helper: re-tokenize a term with TOK_FN and rejoin
    def norm(term: str) -> str:
        return ' '.join(TOK_FN(term))

    manual_terms = load_manual_terms(manual_terms_path, ngram)
    expansion_terms = load_expansion_terms(expansion_terms_path, ngram)

    # Map each term to its normalized form
    manual_norm = {term: norm(term) for term in manual_terms}
    expansion_norm = {term: norm(term) for term in expansion_terms}

    # Universe of all terms to evaluate
    universe = set(manual_terms) | set(expansion_terms)

    # Counters
    TP = FP = TN = FN = 0
    fp_terms = set()
    fn_terms = set()

    for idx, doc in enumerate(docs, start=1):
        tokens = TOK_FN(doc)
        doc_str = ' '.join(tokens)

        # Find which manual terms appear in the document (exact sequence match)
        manual_found = {
            term for term, n in manual_norm.items()
            if n and f" {n} " in f" {doc_str} "
        }
        expansion_found = {
            term for term, n in expansion_norm.items()
            if n and f" {n} " in f" {doc_str} "
        }

        # Evaluate each term in the combined universe
        for term in universe:
            m_hit = term in manual_found
            e_hit = term in expansion_found
            if m_hit and e_hit:
                TP += 1
            elif not m_hit and e_hit:
                FP += 1
                fp_terms.add(term)
            elif m_hit and not e_hit:
                FN += 1
                fn_terms.add(term)
            else:
                TN += 1

        # logging.info(
        #     "Doc %d: manual_found=%d, expansion_found=%d",
        #     idx, len(manual_found), len(expansion_found)
        # )

    # Compute metrics
    total = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    accuracy = (TP + TN) / total if total else 0.0

    # Prepare output file path
    base, _ = os.path.splitext(expansion_terms_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{base}_evaluation_{timestamp}.txt"

    # Write summary and FP/FN lists to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Expansion JSON: {expansion_terms_path}\n")
        if ngram is None:
            f.write("No n-gram filtering applied to manual or expansion terms\n")
        else:
            f.write(f"Filtered manual and expansion terms to {ngram}-gram only\n")
        f.write(f"=== Evaluation Summary Across {len(docs)} Documents ===\n")
        f.write(f"Total terms evaluated: {total}\n")
        f.write(f"True Positives (TP): {TP}\n")
        f.write(f"False Positives (FP): {FP}\n")
        f.write(f"True Negatives (TN): {TN}\n")
        f.write(f"False Negatives (FN): {FN}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n\n")

        if fp_terms:
            f.write(
                f"Unique False Positive terms {len(fp_terms)} "
                f"(expansion predicted but manual not present):\n"
            )
            for t in sorted(fp_terms):
                f.write(f"  {t}\n")
        else:
            f.write("No False Positive terms.\n")

        if fn_terms:
            f.write(
                f"Unique False Negative terms {len(fn_terms)} "
                f"(manual present but expansion missed):\n"
            )
            for t in sorted(fn_terms):
                f.write(f"  {t}\n")
        else:
            f.write("No False Negative terms.\n")

    # Also print summary to console
    print(f"\n=== Evaluation Summary Across {len(docs)} Documents ===")
    if ngram is None:
        print("No n-gram filtering applied to manual or expansion terms")
    else:
        print(f"Filtered manual and expansion terms to {ngram}-gram only")
    print(f"Total terms evaluated: {total}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}\n")

    print(f"\nEvaluation results written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expansion JSON against manual annotations, "
                    "with optional n-gram filtering."
    )
    parser.add_argument(
        "--expansion_path",
        required=True,
        help="Path to the expansion JSON file (e.g. expansions_word2vec_cbow_... .json)."
    )
    parser.add_argument(
        "--manual_dir",
        required=True,
        help=(
            "Directory containing two files:\n"
            "  • manual_terms.txt   (comma-separated manual terms)\n"
            "  • users.txt          (comma-separated list of usernames)"
        )
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        help="If set, use clean_and_tokenize_spellcheck(...) instead of clean_and_tokenize(...)."
    )
    parser.add_argument(
        "--ngram",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Only include manual and expansion terms of this n-gram length (if omitted, accept all lengths)."
    )
    args = parser.parse_args()

    # If spellcheck was requested but not available, abort:
    if args.spellcheck and clean_and_tokenize_spellcheck is None:
        print(
            "ERROR: --spellcheck was requested, but unable to import "
            "`clean_and_tokenize_spellcheck()` from spellchecker_folder/spellchecker.py",
            file=sys.stderr
        )
        sys.exit(1)

    # Choose tokenization function
    global TOK_FN
    if args.spellcheck:
        print("→ Using spell‐checking tokenizer.")
        TOK_FN = clean_and_tokenize_spellcheck
    else:
        print("→ Using vanilla tokenizer (clean_and_tokenize).")
        TOK_FN = clean_and_tokenize

    # Resolve paths to manual_terms.txt and users.txt
    manual_terms_path = os.path.join(args.manual_dir, "manual_terms.txt")
    users_path = os.path.join(args.manual_dir, "users.txt")

    if not os.path.isfile(manual_terms_path):
        parser.error(f"manual_terms.txt not found in {args.manual_dir}")
    if not os.path.isfile(users_path):
        parser.error(f"users.txt not found in {args.manual_dir}")

    # Load users list
    users = load_user_list(users_path)
    if not users:
        parser.error("No users found in users.txt")

    # Fetch documents for those users in r/noburp
    documents = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        filter_users=users
    )
    logging.info("Fetched %d documents for %d users", len(documents), len(users))

    # Run evaluation with (optional) n-gram filtering
    evaluate_terms_performance(
        docs=documents,
        manual_terms_path=manual_terms_path,
        expansion_terms_path=args.expansion_path,
        ngram=args.ngram
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
