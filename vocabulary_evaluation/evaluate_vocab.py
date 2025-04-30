#!/usr/bin/env python3
"""
evaluate_vocab.py

Usage:
    python evaluate_vocab.py /path/to/expanded_vocab.json

Loads the given vocabulary JSON, evaluates it against a set of ground-truth terms and a list of example users,
computes precision, recall, accuracy, Jaccard similarity, and updates the JSON metadata with the evaluation results.
"""
import argparse
import os
import re
import json
from itertools import chain
import sys

# === CONFIG (can be adjusted as needed) ===
GROUND_TRUTHS_PATH = os.path.join("vocabulary_evaluation", "manual_terms.txt")
USERNAMES = [
    "freddiethecalathea",
    "Many_Pomegranate_566",
    "rpesce518",
    "kinglgw",
    "mjh59"
]

# === Imports for evaluation pipeline ===
sys.path.append(os.path.abspath("vocabulary_evaluation"))
sys.path.append(os.path.abspath("."))
from analyze_users import prepare_user_dataframe_multi, preprocess_terms

# === Helper functions ===


def normalize_term(term: str) -> str:
    term = term.lower()
    term = re.sub(r'[^a-z0-9\s\-]', '', term)
    return re.sub(r'\s+', ' ', term).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def match_terms_in_text(terms, text):
    found = set()
    for term in terms:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, text):
            found.add(term)
    return found


def load_ground_truth_terms(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    terms = [t.strip() for t in content.split(',') if t.strip()]
    return [normalize_term(t) for t in preprocess_terms(terms)]


def load_vocab_terms_from_json(vocab_dict):
    raw = list(chain.from_iterable(vocab_dict.values()))
    return set(normalize_term(t) for t in preprocess_terms(raw))


def jaccard_similarity(a, b):
    union = set(a) | set(b)
    if not union:
        return 0.0
    return len(set(a) & set(b)) / len(union)


# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vocabulary JSON against ground truth terms.")
    parser.add_argument('vocab_json', help='Path to the expanded vocabulary JSON file to evaluate')
    args = parser.parse_args()

    vocab_path = args.vocab_json
    if not os.path.exists(vocab_path):
        print(f"Error: vocab file not found: {vocab_path}")
        sys.exit(1)

    # Load vocab JSON
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare term sets
    vocab_dict = data.get('vocabulary', {})
    vocab_terms = load_vocab_terms_from_json(vocab_dict)
    ground_truth_terms = load_ground_truth_terms(GROUND_TRUTHS_PATH)

    # Fetch user texts
    df = prepare_user_dataframe_multi(USERNAMES)
    all_content = " ".join(df['content'].fillna('').tolist())
    normalized_text = normalize_text(all_content)

    # Match terms
    all_possible = set(ground_truth_terms) | vocab_terms
    present = match_terms_in_text(all_possible, normalized_text)

    gt_found = set(ground_truth_terms) & present
    vocab_found = vocab_terms & present
    tp = len(gt_found & vocab_found)
    fn = len(gt_found - vocab_found)
    fp = len(vocab_found - gt_found)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    jaccard = jaccard_similarity(gt_found, vocab_found)

    # Update metadata and write back
    eval_meta = {
        'evaluated_on_users': USERNAMES,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'accuracy': round(accuracy, 4),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'jaccard_similarity': round(jaccard, 4)
    }
    data.setdefault('metadata', {})['evaluation'] = eval_meta

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print results
    print("\n===== Evaluation Metrics =====")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Jaccard Similarity: {jaccard:.2f}\n")

    print("✅ Evaluation complete and written back to JSON.")
