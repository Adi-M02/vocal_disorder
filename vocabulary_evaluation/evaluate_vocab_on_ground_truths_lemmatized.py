import sys
import os
import re
import json
from itertools import chain
import pandas as pd
from pathlib import Path
import spacy
import spacy.cli
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
from spellchecker import SpellChecker
from lemminflect import getAllInflections

# === CONFIG ===
VOCAB_JSON_PATH = "vocab_output_05_21/expanded_output_05_21_15_31_28.json"
GROUND_TRUTHS = "vocabulary_evaluation/manual_terms.txt"
USERNAMES = ["freddiethecalathea", "Many_Pomegranate_566", "rpesce518", "kinglgw", "mjh59"]
PROTECTED_TERMS = {"ibs", "ent", "ents", "gp", "op", "ptsd", "ocd", "rcpd"}

# === SETUP ===
sys.path.append(os.path.abspath("vocabulary_evaluation"))
sys.path.append(os.path.abspath("."))
from analyze_users import prepare_user_dataframe, prepare_user_dataframe_multi, preprocess_terms
# Load the vocabulary JSON
with open(VOCAB_JSON_PATH, "r") as f:
    vocab_data = json.load(f)
# Extract and flatten the list of vocabulary terms
custom_terms = set()
for category_terms in vocab_data["vocabulary"].values():
    for phrase in category_terms:
        # Lowercase and split compound terms (e.g., "Air_vomiting" → ["air", "vomiting"])
        words = phrase.lower().split('_')
        custom_terms.update(words)
spell = SpellChecker()
spell.word_frequency.load_words(custom_terms)
# === UTILITIES ===
def expand_with_inflections(base_word):
    """Return all verb inflections of a word."""
    infl = getAllInflections(base_word, upos='VERB')
    return set(w for forms in infl.values() for w in forms)

def build_term_bins(term_list):
    bins = dict()
    for term in preprocess_terms(term_list):
        lemmatized, cleaned = lemmatize_term(term)
        key = lemmatized
        if key not in bins:
            bins[key] = set()
        bins[key].update({lemmatized, cleaned})
        bins[key].update(expand_with_inflections(lemmatized))
    return bins

def lemmatize_word(word):
    if word in PROTECTED_TERMS:
        return word  # don't touch protected terms
    corrected = spell.correction(word) or word
    token = nlp(corrected)[0]
    lemma = token.lemma_
    return lemma

def lemmatize_term(term):
    term = term.lower()
    term = term.replace('_', ' ')  # Normalize underscores
    term = re.sub(r'[^a-z0-9\s\-]', '', term)
    tokens = re.split(r'\s+', term)

    cleaned_parts = []    # Before lemmatization
    lemmatized_parts = [] # After lemmatization

    for token in tokens:
        if '-' in token:
            subparts = token.split('-')
            cleaned_parts.extend(subparts)
            lemmatized = [lemmatize_word(sub) for sub in subparts]
            lemmatized_parts.extend(lemmatized)
        else:
            cleaned_parts.append(token)
            lemmatized_parts.append(lemmatize_word(token))

    cleaned_term = ' '.join(cleaned_parts)
    lemmatized_term = ' '.join(lemmatized_parts)

    return lemmatized_term, cleaned_term

def normalize_term(term):
    term = term.lower()
    term = re.sub(r'[^a-z0-9\s\-]', '', term)
    return re.sub(r'\s+', ' ', term).strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# def match_terms_in_text(terms, text):
#     found_terms = set()
#     for term in terms:
#         pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
#         if re.search(pattern, text):
#             found_terms.add(term)
#     return found_terms
def match_terms_in_text(term_bins, text):
    found_keys = set()
    for key, variants in term_bins.items():
        for variant in variants:
            pattern = r'(?<!\w)' + re.escape(variant) + r'(?!\w)'
            if re.search(pattern, text):
                found_keys.add(key)
                break
    return found_keys

def load_ground_truth_terms(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        terms = [term.strip() for term in content.split(',') if term.strip()]
    all_terms = set()
    for term in preprocess_terms(terms):
        lemmatized, cleaned = lemmatize_term(term)
        all_terms.add(lemmatized)
        all_terms.add(cleaned)
    return all_terms

def load_vocab_terms_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab_raw = list(chain.from_iterable(vocab_json["vocabulary"].values()))
    # Add both lemmatized and cleaned terms to the set
    all_terms = set()
    for term in preprocess_terms(vocab_raw):
        lemmatized, cleaned = lemmatize_term(term)
        all_terms.add(lemmatized)
        all_terms.add(cleaned)
    return all_terms

if __name__ == "__main__":
    # === LOAD RAW TERMS ===
    with open(VOCAB_JSON_PATH, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab_raw = list(chain.from_iterable(vocab_json["vocabulary"].values()))

    with open(GROUND_TRUTHS, 'r') as f:
        gt_raw = [t.strip() for t in f.read().split(',') if t.strip()]

    # === BUILD TERM BINS ===
    print("Building term bins...")
    vocab_bins = build_term_bins(vocab_raw)
    gt_bins = build_term_bins(gt_raw)

    vocab_terms = set(vocab_bins.keys())
    ground_truth_terms = set(gt_bins.keys())

    # === RUN PIPELINE ===
    print("Preparing user data...")
    df = prepare_user_dataframe_multi(USERNAMES)
    all_text = normalize_text(" ".join(df["content"].fillna("").tolist()))

    # === TERM MATCHING ===
    combined_bins = {**vocab_bins, **gt_bins}
    terms_in_text = match_terms_in_text(combined_bins, all_text)

    matched_gt = ground_truth_terms.intersection(terms_in_text)
    matched_vocab = vocab_terms.intersection(terms_in_text)
    true_positives = matched_gt.intersection(matched_vocab)
    false_negatives = matched_gt - matched_vocab
    false_positives = matched_vocab - matched_gt

    # === METRICS ===
    def jaccard_similarity(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # === ADD METRICS TO METADATA AND WRITE BACK ===
    vocab_json["metadata"].update({
        "evaluation": {
            "evaluated_on_users": USERNAMES,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1_score, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "jaccard_similarity": round(jaccard_similarity(matched_gt, matched_vocab), 4)
        }
    })
    path = Path(VOCAB_JSON_PATH)
    lemmatized_path = path.with_name(f"{path.stem}_lemmatized{path.suffix}")
    with open(lemmatized_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)

    # === PRINT RESULTS ===
    print("\n===== Evaluation Metrics =====")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    print("\n===== Term Match Breakdown =====")
    print("\n✅ True Positives:", sorted(true_positives))
    print("\n❌ Missed by Vocab (False Negatives):", sorted(false_negatives))
    print("\n⚠️ False Positives:", sorted(false_positives))
    print("\nJaccard Similarity:", jaccard_similarity(matched_gt, matched_vocab))

    not_in_text = ground_truth_terms - matched_gt
    print("\n🔍 Ground truth terms NOT found in the text:")
    print(sorted(not_in_text))
    print(f"Count: {len(not_in_text)}")

    print(f"\n✅ Updated JSON with evaluation metrics saved to {lemmatized_path}")
    def save_term_bins_combined(vocab_bins, gt_bins, script_path):
        output_path = Path(script_path).with_name("lemmatization_testing.json")
        combined = {
            "vocab_bins": {key: sorted(list(val)) for key, val in vocab_bins.items()},
            "ground_truth_bins": {key: sorted(list(val)) for key, val in gt_bins.items()}
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

    save_term_bins_combined(vocab_bins, gt_bins, __file__)
