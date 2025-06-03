#!/usr/bin/env python3
import sys, os, json, argparse, logging
from datetime import datetime
from typing import List, Tuple, Optional, Counter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords

sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize
from query_mongo import return_documents

# ─────────────────────────────────────────────────────────────
# Helper: embed any phrase (1- or 2-tokens) by mean-pooling
# ─────────────────────────────────────────────────────────────
def embed_phrase(model: Word2Vec, phrase: str) -> Optional[np.ndarray]:
    tokens = clean_and_tokenize(phrase)
    vecs   = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    return None if not vecs else np.mean(vecs, axis=0)

# ─────────────────────────────────────────────────────────────
# Helper: extract frequent bigrams from Mongo, filter stop-words
# ─────────────────────────────────────────────────────────────
def extract_frequent_bigrams(
    min_count: int,
    db_name: str,
    collection_name: str,
    filter_subreddits: Optional[List[str]] = None,
    mongo_uri: str = "mongodb://localhost:27017/"
) -> List[Tuple[str, str]]:
    docs = return_documents(db_name, collection_name,
                            filter_subreddits, mongo_uri=mongo_uri)
    from collections import Counter
    bigram_counts: Counter[Tuple[str, str]] = Counter()
    for doc in docs:
        tokens = clean_and_tokenize(doc)
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[(w1, w2)] += 1

    stopset = set(stopwords.words("english"))
    return [
        bigram for bigram, cnt in bigram_counts.items()
        if cnt >= min_count and bigram[0] not in stopset
                           and bigram[1] not in stopset
    ]

# ─────────────────────────────────────────────────────────────
# Build phrase list & embedding matrix (unigrams + bigrams)
# ─────────────────────────────────────────────────────────────
def build_phrase_embeddings(
    model: Word2Vec,
    bigrams: List[Tuple[str, str]],
    exclude_terms: set[str] | None = None
):
    phrase_list: List[str] = []
    emb_rows     : List[np.ndarray] = []

    # 1) unigrams straight from the model vocab
    for tok in model.wv.key_to_index.keys():
        if exclude_terms and tok in exclude_terms:
            continue
        phrase_list.append(tok)
        emb_rows.append(model.wv[tok])

    # 2) space-joined bigrams (token_a token_b)
    for w1, w2 in bigrams:
        if w1 not in model.wv.key_to_index or w2 not in model.wv.key_to_index:
            continue                           # skip OOV components
        phrase = f"{w1} {w2}"
        if exclude_terms and phrase in exclude_terms:
            continue
        phrase_list.append(phrase)
        emb_rows.append((model.wv[w1] + model.wv[w2]) / 2)

    emb_matrix = np.vstack(emb_rows)
    norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.clip(norms, 1e-9, None)  # unit-norm
    return phrase_list, emb_matrix

# ─────────────────────────────────────────────────────────────
# Neighbour search (handles uni- *and* bi-gram queries)
# ─────────────────────────────────────────────────────────────
def find_vocab_neighbors(
    model: Word2Vec,
    phrase_list: List[str],
    model_label: str,
    query_phrase: str,
    top_k: int,
    eps: Optional[float] = None
):
    # -------- 0. unit-norm query vector --------
    q_vec = embed_phrase(model, query_phrase)
    if q_vec is None:
        print(f"[{model_label}] Some query tokens are OOV → {query_phrase}")
        return
    q_vec /= max(np.linalg.norm(q_vec), 1e-9)

    # -------- 1. similarity for every phrase ---
    sims: List[float] = []
    for phrase in phrase_list:
        if " " not in phrase:                         # unigram
            sims.append(np.dot(model.wv[phrase] /
                               np.linalg.norm(model.wv[phrase]), q_vec))
        else:                                         # bigram
            t1, t2 = phrase.split()
            sim1 = np.dot(model.wv[t1] / np.linalg.norm(model.wv[t1]), q_vec)
            sim2 = np.dot(model.wv[t2] / np.linalg.norm(model.wv[t2]), q_vec)
            sims.append(0.5 * (sim1 + sim2))          # arithmetic mean

    sims = np.asarray(sims)

    # -------- 2. choose indices -----------------
    if eps is not None:
        keep = np.where(sims >= eps)[0]
        # sort by similarity descending
        idx_order = keep[np.argsort(-sims[keep])]
        header = f"Neighbours with sim ≥ {eps}"
    else:
        idx_order = np.argsort(-sims)[:top_k]
        header = f"Top {top_k} neighbours"

    # -------- 3. print nicely -------------------
    print(f"\n[{model_label}] {header} for '{query_phrase}':")
    for idx in idx_order:
        phrase = phrase_list[idx]
        sim    = sims[idx]
        if " " not in phrase:
            print(f"  {phrase:<25} sim {sim:.4f}")
        else:
            t1, t2 = phrase.split()
            sim1 = np.dot(model.wv[t1] / np.linalg.norm(model.wv[t1]), q_vec)
            sim2 = np.dot(model.wv[t2] / np.linalg.norm(model.wv[t2]), q_vec)
            print(f"  {phrase:<25} avg {sim:.4f} ; {t1}:{sim1:.4f} , {t2}:{sim2:.4f}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query nearest neighbours (uni + frequent bi-grams).")
    parser.add_argument("-d", "--model_dir", required=True,
                        help="Directory containing .model files")
    parser.add_argument("--db", default="reddit", help="MongoDB database")
    parser.add_argument("--coll", default="noburp_all",
                        help="MongoDB collection")
    parser.add_argument("--filter_subreddits", default=["noburp"],
                        help="Comma-sep list; restrict bigram mining")
    parser.add_argument("--min_bigram_count", type=int, default=5,
                        help="Frequency cut-off for bigrams")
    parser.add_argument("--exclude_terms_json", default=None,
                        help="JSON of phrases to exclude entirely")
    parser.add_argument("--query", required=True,
                        help="Unigram or bigram to query")
    parser.add_argument("-k", type=int, default=15,
                        help="Number of neighbours to show")
    parser.add_argument("--eps", type=float, default=None,
                    help="Cosine-sim threshold; return *all* phrases with sim ≥ eps")
    args = parser.parse_args()

    fsr = (args.filter_subreddits
           if args.filter_subreddits else None)

    # optional exclusion set (e.g. rcpd_terms.json)
    exclude_set = set()
    if args.exclude_terms_json:
        with open(args.exclude_terms_json, encoding="utf-8") as f:
            for cat, terms in json.load(f).items():
                exclude_set.update(term.lower().strip() for term in terms)

    # Mine bigrams once (shared by both models)
    bigrams = extract_frequent_bigrams(args.min_bigram_count,
                                       args.db, args.coll, fsr)

    for fname, label in [("word2vec_cbow.model",  "CBOW"),
                         ("word2vec_skipgram.model", "SkipGram")]:
        path = os.path.join(args.model_dir, fname)
        print(f"\n[{label}] loading {path} …")
        model = Word2Vec.load(path)

        phrase_list, emb_matrix = build_phrase_embeddings(
            model, bigrams, exclude_set
        )

        find_vocab_neighbors(
            model, phrase_list,
            label, args.query,
            top_k=args.k,
            eps=args.eps
        )

    print("\nDone.")
