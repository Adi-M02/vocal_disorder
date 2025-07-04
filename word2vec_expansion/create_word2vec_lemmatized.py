"""
usage python word2vec_expansion/create_word2vec_spellcheck.py

Train Word2Vec on Reddit posts with optional spell-checking and lemmatization lookup.

This version:
  • Adds lemmatization via a precomputed lookup table before training.
  • Supports both vanilla and spell-checked tokenization.
  • Injects custom domain terms into the vocab.
"""
import sys
import os
import datetime
import time
import json
import argparse
from pathlib import Path

from gensim.models import Word2Vec

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list


def load_terms(path: str) -> dict[str, list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }


def load_lookup(path: str) -> dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on r/noburp Reddit posts with lemmatization lookup."
    )
    parser.add_argument(
        "--spellcheck", action="store_true",
        help="Use clean_and_tokenize_spellcheck instead of clean_and_tokenize."
    )
    parser.add_argument(
        "--lookup", type=str,
        default="testing/lemma_lookup.json",
        help="Path to JSON lemma lookup table."
    )
    parser.add_argument(
        "--vector_size", type=int, default=300,
        help="Embedding dimensionality."
    )
    parser.add_argument(
        "--window", type=int, default=7,
        help="Context window size."
    )
    parser.add_argument(
        "--min_count", type=int, default=3,
        help="Ignore tokens with total frequency lower than this."
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory to save models and info."
    )
    args = parser.parse_args()

    # Load lemma lookup map
    lookup_map = load_lookup(args.lookup)
    logging = print  # simple print for progress

    # Choose tokenizer
    if args.spellcheck:
        print("Using spell-checking tokenizer")
        def token_fn(text):
            return spellcheck_token_list(clean_and_tokenize(text))
    else:
        print("Using vanilla tokenizer")
        token_fn = clean_and_tokenize

    # 2) Fetch raw Reddit docs
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    logging(f"Number of documents fetched: {len(docs)}")

    # 3) Tokenize + apply lookup
    cleaned_docs = []
    for text in docs:
        toks = token_fn(text)
        # apply lemma lookup
        lemtoks = [ lookup_map.get(tok, tok) for tok in toks ]
        cleaned_docs.append(lemtoks)
    logging(f"Tokenized & lemmatized {len(cleaned_docs)} documents")

    # 5) Build output directory
    now = datetime.datetime.now()
    base_outdir = Path(args.outdir) if args.outdir else Path("word2vec_expansion")
    out_dir = base_outdir / now.strftime("word2vec_%m_%d_%H_%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) Save run info
    info = {**vars(args), "timestamp": now.isoformat()}
    with open(out_dir / "info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    # 7) Train CBOW
    start_cb = time.time()
    cbow = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=0,
        workers=max(1, os.cpu_count()-1)
    )
    cbow.build_vocab(cleaned_docs)
    cbow.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=5)
    cbow_path = out_dir / "word2vec_cbow.model"
    cbow.save(str(cbow_path))
    logging(f"CBOW training took {time.time()-start_cb:.2f}s → saved to {cbow_path}")

    # 8) Train Skip-gram
    start_sg = time.time()
    skipgram = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=1,
        workers=max(1, os.cpu_count()-1)
    )
    skipgram.build_vocab(cleaned_docs)
    skipgram.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=5)
    skipgram_path = out_dir / "word2vec_skipgram.model"
    skipgram.save(str(skipgram_path))
    logging(f"Skip-gram training took {time.time()-start_sg:.2f}s → saved to {skipgram_path}")

if __name__ == "__main__":
    main()
