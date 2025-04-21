import sys
import os
import re
import json
from itertools import chain
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# === CONFIG ===
VOCAB_JSON_PATH = "vocab_output_04_20/expanded_vocab_wordnet_top60_Bio_ClinicalBERT.json"
GROUND_TRUTHS = "vocabulary_evaluation/manual_terms.txt"
USERNAMES = ["freddiethecalathea", "Many_Pomegranate_566", "rpesce518"]

# === SETUP ===
sys.path.append(os.path.abspath("vocabulary_evaluation"))
sys.path.append(os.path.abspath("."))
from analyze_users import prepare_user_dataframe_multi, preprocess_terms

# === NLTK Setup ===
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    from nltk import pos_tag
    tag = pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

# === UTILITIES ===
def normalize_term(term):
    term = term.lower()
    term = re.sub(r'[^a-z0-9\s\-]', '', term)
    return re.sub(r'\s+', ' ', term).strip()

def lemmatize_vocab_term(term):
    tokens = term.lower().split("_")
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    return "_".join(lemmatized)

def lemmatize_plain_term(term):
    tokens = word_tokenize(term.lower())
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    return " ".join(lemmatized)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def match_terms_in_text(terms, text):
    found_terms = set()
    for term in terms:
        pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
        if re.search(pattern, text):
            found_terms.add(term)
    return found_terms

def load_ground_truth_terms(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        terms = [term.strip() for term in content.split(',') if term.strip()]
        processed = preprocess_terms(terms)
        return [lemmatize_plain_term(normalize_term(term)) for term in processed]

def load_vocab_terms_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab_raw = list(chain.from_iterable(vocab_json["vocabulary"].values()))
    processed = preprocess_terms(vocab_raw)
    return set(lemmatize_vocab_term(normalize_term(term)) for term in processed)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    vocab_terms = load_vocab_terms_from_json(VOCAB_JSON_PATH)
    ground_truth_terms = load_ground_truth_terms(GROUND_TRUTHS)

    df = prepare_user_dataframe_multi(USERNAMES)
    all_text = normalize_text(" ".join(df["content"].fillna("").tolist()))

    all_possible_terms = set(ground_truth_terms).union(vocab_terms)
    terms_in_text = match_terms_in_text(all_possible_terms, all_text)

    matched_gt = set(ground_truth_terms).intersection(terms_in_text)
    matched_vocab = set(vocab_terms).intersection(terms_in_text)
    true_positives = matched_gt.intersection(matched_vocab)
    false_negatives = matched_gt - matched_vocab
    false_positives = matched_vocab - matched_gt

    def jaccard_similarity(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    with open(VOCAB_JSON_PATH, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    vocab_json["metadata"].update({
        "evaluation": {
            "evaluated_on_users": USERNAMES,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "jaccard_similarity": round(jaccard_similarity(matched_gt, matched_vocab), 4)
        }
    })

    with open(VOCAB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)

    print("\n===== Evaluation Metrics =====")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    print("\n===== Term Match Breakdown =====")
    print("\n✅ True Positives:", sorted(true_positives))
    print("\n❌ Missed by Vocab (False Negatives):", sorted(false_negatives))
    print("\n⚠️ False Positives:", sorted(false_positives))
    print("\nJaccard Similarity:", jaccard_similarity(matched_gt, matched_vocab))

    not_in_text = set(ground_truth_terms) - matched_gt
    print("\n🔍 Ground truth terms NOT found in the text:")
    print(sorted(not_in_text))
    print(f"Count: {len(not_in_text)}")
    for term in sorted(not_in_text):
        if term in all_text:
            print(f"✅ Found '{term}' in normalized text")
        else:
            print(f"❌ '{term}' still not found")

    print(f"\n✅ Updated JSON with evaluation metrics saved to {VOCAB_JSON_PATH}")
