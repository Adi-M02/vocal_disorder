"""
usage: python word2vec_expansion/evaluate_embedding_space.py [options]
"""
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
    q_vec = embed_phrase(model, query_phrase)
    if q_vec is None:
        print(f"[{model_label}] Some query tokens are OOV → {query_phrase}")
        return

    dists: List[float] = []
    for phrase in phrase_list:
        if " " not in phrase:  # unigram
            vec = model.wv[phrase]
            dist = np.linalg.norm(q_vec - vec)
        else:  # bigram
            t1, t2 = phrase.split()
            if t1 in model.wv and t2 in model.wv:
                vec = (model.wv[t1] + model.wv[t2]) / 2
                dist = np.linalg.norm(q_vec - vec)
            else:
                dist = float("inf")  # if component is OOV
        dists.append(dist)

    dists = np.asarray(dists)

    if eps is not None:
        keep = np.where(dists <= eps)[0]
        idx_order = keep[np.argsort(dists[keep])]
        header = f"Neighbours with distance ≤ {eps}"
    else:
        idx_order = np.argsort(dists)[:top_k]
        header = f"Top {top_k} closest (by Euclidean distance)"

    print(f"\n[{model_label}] {header} for '{query_phrase}':")
    for idx in idx_order:
        phrase = phrase_list[idx]
        dist = dists[idx]
        print(f"  {phrase:<25} dist {dist:.4f}")

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
    parser.add_argument("--min_bigram_count", type=int, default=0,
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
    bigrams = []

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
