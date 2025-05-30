#!/usr/bin/env python3
import os
import sys

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from transformers import AutoTokenizer

# Four models to check
MODELS = [
    ("bert-base",      "bert-base-uncased"),
    ("bertweet",       "vinai/bertweet-base"),
    ("clinical-bert",  "emilyalsentzer/Bio_ClinicalBERT"),
    ("pubmed-bert",    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
]

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fetch_unique_terms():
    """
    Fetch raw documents as text, clean/tokenize them,
    and return a set of all unique tokens.
    """
    raw_texts = return_documents("reddit", "noburp_all", ["noburp"])
    unique_terms = set()
    for doc in raw_texts:
        if not isinstance(doc, str):
            continue
        tokens = clean_and_tokenize(doc)
        unique_terms.update(tokens)
    print(f"Total unique terms found: {len(unique_terms)}")
    return unique_terms


def check_coverage(unique_terms):
    """
    For each model's tokenizer, report which terms map to <unk>.
    """
    total = len(unique_terms)
    for name, checkpoint in MODELS:
        print(f"\n=== Coverage for {name} ({checkpoint}) ===")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        unk_id = tokenizer.unk_token_id
        missing = [term for term in unique_terms if tokenizer.convert_tokens_to_ids(term) == unk_id]
        print(f"Missing {len(missing)}/{total} tokens ({len(missing)/total*100:.2f}%)")
        if missing:
            print("Sample missing tokens:", missing[:20])
        else:
            print("All terms recognized.")


def main():
    print("Fetching and tokenizing dataset docs...")
    unique_terms = fetch_unique_terms()
    print(f"Total unique terms in dataset: {len(unique_terms)}")

    print("\nChecking tokenizer coverage for each model...")
    check_coverage(unique_terms)
    print("\nDone.")

if __name__ == "__main__":
    main()
