import sys
import os
import re
import json
from itertools import chain
import pandas as pd

# === CONFIG ===
VOCAB_JSON_PATH = "vocab_output_04_18/expanded_vocab_ngram_top20_uf5_bf3_tf2.json"
GROUND_TRUTHS = "vocabulary_evaluation/manual_terms.txt"
USERNAME = "freddiethecalathea"

# === SETUP ===
sys.path.append(os.path.abspath("vocabulary_evaluation"))
sys.path.append(os.path.abspath("."))
from analyze_users import prepare_user_dataframe, preprocess_terms

# === UTILITIES ===
def normalize_term(term):
    term = term.lower()
    term = re.sub(r'[^a-z0-9\s\-]', '', term)
    return re.sub(r'\s+', ' ', term).strip()

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
        return [normalize_term(term) for term in preprocess_terms(terms)]

def load_vocab_terms_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab_raw = list(chain.from_iterable(vocab_json["vocabulary"].values()))
    return set(normalize_term(term) for term in preprocess_terms(vocab_raw))

# === LOAD TERMS ===
ground_truth_terms = load_ground_truth_terms(GROUND_TRUTHS)
vocab_terms = load_vocab_terms_from_json(VOCAB_JSON_PATH)

# === RUN PIPELINE ===
df = prepare_user_dataframe(USERNAME)
# df[["subreddit", "content"]].to_csv("test.csv", index=False)
all_text = normalize_text(" ".join(df["content"].fillna("").tolist()))

# === TERM MATCHING ===
all_possible_terms = set(ground_truth_terms).union(vocab_terms)
terms_in_text = match_terms_in_text(all_possible_terms, all_text)

# === TERM MATCH ANALYSIS ===
matched_gt = set(ground_truth_terms).intersection(terms_in_text)
matched_vocab = set(vocab_terms).intersection(terms_in_text)

true_positives = matched_gt.intersection(matched_vocab)
false_negatives = matched_gt - matched_vocab
false_positives = matched_vocab - matched_gt

# === METRICS ===
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

tp = len(true_positives)
fp = len(false_positives)
fn = len(false_negatives)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

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
