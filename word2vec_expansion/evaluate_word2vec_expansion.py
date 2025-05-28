"""
Script to load Word2Vec expansion terms, manual terms, fetch user documents,
clean and tokenize them, and evaluate expansion performance against manual terms
by computing TP, FP, TN, FN per document, then precision, recall, accuracy.
Logs false positives and false negatives.
Handles multi-word phrases by exact token-sequence matching.
"""
import sys
import json
import logging
from collections import Counter

# Adjust path so we can import the user's tokenizer
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize

# Import return_documents from wherever it's defined; adjust the module path as needed
from query_mongo import return_documents


def load_expansion_terms(path: str) -> list[str]:
    """
    Load expansion terms from a JSON file and flatten values into a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    terms: list[str] = []
    for value in data.values():
        if isinstance(value, list):
            terms.extend(value)
        else:
            terms.append(value)
    return terms


def load_manual_terms(path: str) -> list[str]:
    """
    Load comma-separated manual terms from a TXT file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [t.strip() for t in content.split(',') if t.strip()]


def evaluate_terms_performance(
    docs: list[str],
    manual_terms: list[str],
    expansion_terms: list[str]
) -> None:
    """
    For each document, tokenize and detect manual vs expansion terms.
    Computes TP, FP, TN, FN across each term in the union of manual and expansion sets,
    then aggregates totals, calculates precision, recall, accuracy,
    and logs false positive/negative term lists.
    """
    # Normalize term lists to tokenized strings for matching
    def norm(term: str) -> str:
        return ' '.join(clean_and_tokenize(term))

    manual_norm = {term: norm(term) for term in manual_terms}
    expansion_norm = {term: norm(term) for term in expansion_terms}

    # Universe of all terms to evaluate
    universe = set(manual_terms) | set(expansion_terms)

    # Initialize counters and term trackers
    TP = FP = TN = FN = 0
    fp_terms = set()
    fn_terms = set()

    for idx, doc in enumerate(docs, start=1):
        tokens = clean_and_tokenize(doc)
        doc_str = ' '.join(tokens)

        # Determine which original terms are found
        manual_found = {
            term for term, n in manual_norm.items()
            if n and f" {n} " in f" {doc_str} "
        }
        expansion_found = {
            term for term, n in expansion_norm.items()
            if n and f" {n} " in f" {doc_str} "
        }

        # Evaluate per-term
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

        logging.info(
            "Doc %d: manual_found=%d, expansion_found=%d",
            idx, len(manual_found), len(expansion_found)
        )

    # Compute metrics
    total = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    accuracy = (TP + TN) / total if total else 0.0

    # Print summary
    print(f"\n=== Evaluation Summary Across {len(docs)} Documents ===")
    print(f"Total terms evaluated: {total}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}\n")

    # Log FP and FN term lists
    if fp_terms:
        print("False Positive terms (expansion predicted but manual not present):")
        for t in sorted(fp_terms):
            print(f"  {t}")
    else:
        print("No False Positive terms.")

    if fn_terms:
        print("False Negative terms (manual present but expansion missed):")
        for t in sorted(fn_terms):
            print(f"  {t}")
    else:
        print("No False Negative terms.")


def main():
    # File paths
    expansion_path = "word2vec_expansion/word2vec_05_26_16_04/expansions_word2vec_cbow_05_27_17_50.json"
    manual_terms_path = "vocabulary_evaluation/manual_terms.txt"

    # Load terms
    expansion_terms = load_expansion_terms(expansion_path)
    logging.info("Loaded %d expansion terms", len(expansion_terms))

    manual_terms = load_manual_terms(manual_terms_path)
    logging.info("Loaded %d manual terms", len(manual_terms))

    # Users to fetch
    users = ["freddiethecalathea", "Many_Pomegranate_566", "rpesce518", "kinglgw", "mjh59"]

    # Fetch docs
    documents = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        filter_users=users
    )
    logging.info("Fetched %d documents", len(documents))

    # Evaluate performance
    evaluate_terms_performance(documents, manual_terms, expansion_terms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()