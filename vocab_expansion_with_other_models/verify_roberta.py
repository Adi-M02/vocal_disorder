#!/usr/bin/env python3
import os
import sys

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM

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


def test_bertweet_add_tokens(unique_terms):
    checkpoint = "vinai/bertweet-base"
    print(f"\nLoading tokenizer and model for {checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    # Identify missing tokens
    unk_id = tokenizer.unk_token_id
    missing = [t for t in unique_terms if tokenizer.convert_tokens_to_ids(t) == unk_id]
    print(f"Missing tokens: {len(missing)}/{len(unique_terms)} ({len(missing)/len(unique_terms)*100:.2f}%)")
    if len(missing) > 0:
        print("Sample missing:", missing[:20])

    # Attempt to add missing tokens
    try:
        print("\nAdding missing tokens as additional_special_tokens...")
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        vocab_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.size(0)
        print(f"After addition → tokenizer vocab size = {vocab_size}, embedding rows = {emb_size}")
        assert vocab_size == emb_size, "Vocab and embedding sizes do not match!"
        print("Token addition succeeded: tokenizer and model are aligned.")
    except Exception as e:
        print("Error during token addition:", repr(e))
        sys.exit(1)


def main():
    terms = fetch_unique_terms()
    test_bertweet_add_tokens(terms)

if __name__ == "__main__":
    main()
