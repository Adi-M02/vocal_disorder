"""
usage: python word2vec_expansion/evaluate_embedding_space.py [options]
"""
import sys, os, json, argparse, logging
from datetime import datetime
from typing import List, Tuple, Optional, Counter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from nltk.corpus import stopwords

sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize
from query_mongo import return_documents
from spellchecker_folder.spellchecker import spellcheck_token_list

# ─────────────────────────────────────────────────────────────
# Helper: embed any phrase (1- or 2-tokens) by mean-pooling
# ─────────────────────────────────────────────────────────────
def embed_phrase(model: Word2Vec, phrase: str) -> Optional[np.ndarray]:
    tokens = clean_and_tokenize(phrase)
    vecs   = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    return None if not vecs else np.mean(vecs, axis=0)

# ─────────────────────────────────────────────────────────────
# Helper: extract frequent n-grams
# ─────────────────────────────────────────────────────────────
def extract_frequent_ngrams(
    max_ngram: 3,
    query: str,
    tok_fn,
    lookup_map: dict
) -> list[str]:
    # fetch all documents
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        mongo_uri="mongodb://localhost:27017/"
    )
    
    counts = Counter()
    
    # slide an n-length window over each doc’s token list
    if max_ngram <= 1:
        return []
    for doc in tqdm(docs, desc=f"Loading docs for ngrams"):
        tokens = [lookup_map.get(t, t) for t in tok_fn(doc)]
        L = len(tokens)
        for n in range(2, max_ngram + 1):
            if L < n:
                break
            for i in range(L - n + 1):
                if any(term for term in query.split() in tokens[i:i + n]):
                    continue
                gram = tuple(tokens[i:i + n])
                counts[gram] += 1

    # hardcoded min_count for 2-gram and 3-gram
    result = []
    for gram, cnt in counts.items():
        n = len(gram)
        if n == 2 and cnt >= 1:
            result.append(" ".join(gram))
        elif n == 3 and cnt >= 1:
            result.append(" ".join(gram))
    return result

# ─────────────────────────────────────────────────────────────
# Build phrase list & embedding matrix (unigrams + ngrams)
# ─────────────────────────────────────────────────────────────
def build_phrase_embeddings(
    model: Word2Vec,
    ngrams: List[str]
) -> tuple[List[str], np.ndarray]:
    phrase_list: List[str] = []
    emb_rows: List[np.ndarray] = []

    # 1) unigrams straight from the model vocab
    for tok in model.wv.key_to_index.keys():
        phrase_list.append(tok)
        emb_rows.append(model.wv[tok])

    # 2) space-joined ngrams (each ngram is a string like "token_a token_b")
    for phrase in ngrams:
        tokens = phrase.split()
        # skip if any component is out-of-vocab
        if any(tok not in model.wv.key_to_index for tok in tokens):
            continue
        phrase_list.append(phrase)
        # average the embeddings of each token
        emb_rows.append(np.mean([model.wv[tok] for tok in tokens], axis=0))

    # stack into matrix and normalize to unit length
    emb_matrix = np.vstack(emb_rows)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.clip(norms, 1e-9, None)

    return phrase_list, emb_matrix

# ─────────────────────────────────────────────────────────────
# Neighbour search (handles uni- *and* n gram queries)
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
        tokens = phrase.split()
        # skip if any token is OOV
        if any(t not in model.wv.key_to_index for t in tokens):
            sims.append(-1.0)
            continue
        # mean of normalized token vectors
        phrase_vec = np.mean([model.wv[t] / np.linalg.norm(model.wv[t]) for t in tokens], axis=0)
        sim = np.dot(phrase_vec, q_vec)
        sims.append(sim)

    sims = np.asarray(sims)

    # -------- 2. choose indices -----------------
    if eps is not None:
        keep = np.where(sims >= eps)[0]
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
        tokens = phrase.split()
        if len(tokens) == 1:
            print(f"  {phrase:<25} sim {sim:.4f}")
        else:
            # show sim for each token as well as mean
            token_sims = [np.dot(model.wv[t] / np.linalg.norm(model.wv[t]), q_vec) for t in tokens]
            token_str = " , ".join(f"{t}:{s:.4f}" for t, s in zip(tokens, token_sims))
            print(f"  {phrase:<25} avg {sim:.4f} ; {token_str}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query nearest neighbours (uni + frequent n-grams).")
    parser.add_argument("-d", "--model_dir", required=True,
                        help="Directory containing .model files")
    parser.add_argument("--ngram_count", type=int, default=3)
    parser.add_argument("--db", default="reddit", help="MongoDB database")
    parser.add_argument("--coll", default="noburp_all",
                        help="MongoDB collection")
    parser.add_argument("--filter_subreddits", default=["noburp"],
                        help="Comma-sep list; restrict ngram mining")
    parser.add_argument("--query", required=True,
                        help="Unigram or ngram to query")
    parser.add_argument("-k", type=int, default=15,
                        help="Number of neighbours to show")
    parser.add_argument("--eps", type=float, default=None,
                    help="Cosine-sim threshold; return *all* phrases with sim ≥ eps")
    args = parser.parse_args()

    fsr = (args.filter_subreddits
           if args.filter_subreddits else None)

    def tok_fn(text):
        return spellcheck_token_list(clean_and_tokenize(text))
    # Load lemma lookup map
    lookup = json.load(open("testing/lemma_lookup.json", "r", encoding="utf-8"))
    # Mine ngrams once (shared by both models)
    ngrams = extract_frequent_ngrams(args.ngram_count, args.query, tok_fn, lookup)

    for fname, label in [("word2vec_cbow.model",  "CBOW"),
                         ("word2vec_skipgram.model", "SkipGram")]:
        path = os.path.join(args.model_dir, fname)
        print(f"\n[{label}] loading {path} …")
        model = Word2Vec.load(path)

        phrase_list, emb_matrix = build_phrase_embeddings(
            model, ngrams
        )

        find_vocab_neighbors(
            model, phrase_list,
            label, args.query,
            top_k=args.k,
            eps=args.eps
        )

    print("\nDone.")
