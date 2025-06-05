import sys
import os
import datetime
import time
import json
import argparse

from gensim.models import Word2Vec

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import clean_and_tokenize_spellcheck
import random


def load_terms(path: str) -> dict[str, list[str]]:
    """
    Load category→list-of-terms from JSON, replacing underscores with spaces.
    """
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on r/noburp Reddit posts, with optional spell-checking."
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        help="If set, use clean_and_tokenize_spellcheck(...) instead of clean_and_tokenize(...)."
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=300,
        help="Embedding dimensionality (default: 300)."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Maximum distance between target word and context words (default: 7)."
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Ignores all tokens with total frequency lower than this (default: 5)."
    )
    args = parser.parse_args()

    # If user asked for spellcheck but the function isn’t available, abort:
    if args.spellcheck and clean_and_tokenize_spellcheck is None:
        print(
            "ERROR: --spellcheck was requested, but unable to import "
            "`clean_and_tokenize_spellcheck()` from spellchecker_folder/spellchecker.py",
            file=sys.stderr
        )
        sys.exit(1)

    # Decide which tokenizer to use:
    if args.spellcheck:
        print("→ Using spell-checking tokenizer.")
        token_fn = clean_and_tokenize_spellcheck
    else:
        print("→ Using vanilla tokenizer (clean_and_tokenize).")
        token_fn = clean_and_tokenize

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Fetch raw Reddit documents (as strings)
    # ─────────────────────────────────────────────────────────────────────────────
    docs = return_documents("reddit", "noburp_all", ["noburp"])
    print(f"Number of documents fetched from MongoDB: {len(docs)}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Clean + tokenize (or clean+spellcheck+tokenize) each document
    # ─────────────────────────────────────────────────────────────────────────────
    cleaned_docs: list[list[str]] = []
    for doc in docs:
        toks = token_fn(doc)
        cleaned_docs.append(toks)
    print(f"Number of tokenized documents: {len(cleaned_docs)}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Load & tokenize custom terms so they get injected into the W2V vocab
    # ─────────────────────────────────────────────────────────────────────────────
    custom_terms_map = load_terms("rcpd_terms.json")
    custom_token_lists: list[list[str]] = []

    for category, terms in custom_terms_map.items():
        for term in terms:
            tok_list = token_fn(term)
            if tok_list:
                custom_token_lists.append(tok_list)

        # Also inject the category name itself as tokens
        cat_toks = token_fn(category)
        if cat_toks:
            custom_token_lists.append(cat_toks)

    total_custom_terms = sum(len(v) for v in custom_terms_map.values())
    print(
        f"Loaded {total_custom_terms} custom‐category terms → "
        f"{len(custom_token_lists)} token‐lists for custom vocab insertion"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # 5) Build a timestamped output directory
    # ─────────────────────────────────────────────────────────────────────────────
    now = datetime.datetime.now()
    out_dir = now.strftime("word2vec_expansion/word2vec_%m_%d_%H_%M")
    os.makedirs(out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # 6) Write an info JSON file under out_dir with all argument values
    # ─────────────────────────────────────────────────────────────────────────────
    info = {
        "spellcheck": args.spellcheck,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
    }
    info_path = os.path.join(out_dir, "info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote parameter info to {info_path}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 7) Train CBOW (with custom‐vocab update), using the provided hyperparameters
    # ─────────────────────────────────────────────────────────────────────────────
    start_cbow = time.time()
    cbow = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=0,  # CBOW
        workers=max(1, os.cpu_count() - 4)
    )
    cbow.build_vocab(cleaned_docs)
    cbow.build_vocab(custom_token_lists * 5, update=True)
    cbow.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=5)
    cbow_path = os.path.join(out_dir, "word2vec_cbow.model")
    cbow.save(cbow_path)
    print(f"CBOW training took {time.time() - start_cbow:.2f} seconds → saved to {cbow_path}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 8) Train Skip-gram (with custom‐vocab update), using the same hyperparameters
    # ─────────────────────────────────────────────────────────────────────────────
    start_skip = time.time()
    skipgram = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=1,  # Skip-gram
        workers=max(1, os.cpu_count() - 4)
    )
    skipgram.build_vocab(cleaned_docs)
    skipgram.build_vocab(custom_token_lists * 5, update=True)
    skipgram.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=5)
    skipgram_path = os.path.join(out_dir, "word2vec_skipgram.model")
    skipgram.save(skipgram_path)
    print(f"Skip-gram training took {time.time() - start_skip:.2f} seconds → saved to {skipgram_path}")

    print(f"All models saved under {out_dir}")


if __name__ == "__main__":
    main()