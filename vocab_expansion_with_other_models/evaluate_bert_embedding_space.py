"""
──────────────────────────────────────────────────────────────────────
• Mines unigrams & bigrams from a Mongo corpus (clean_and_tokenize)
• Keeps terms whose frequency ≥ --min_count  (stop-words skipped)
• Embeds every unigram with a HuggingFace BERT model
• Bigram score  =  1/2*(cos(query, w1) + cos(query, w2))
• Supports either   --k  N   or   --eps  τ   (ε wins if both passed)
usage: python vocab_expansion_with_other_models/evaluate_bert_embedding_space.py [options]
"""

import os, sys, json, argparse, logging, math
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from collections import Counter

sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize
from query_mongo import return_documents

# ─────────────────────────────────────────────────────────────
# 1. Mine unigrams & bigrams with frequency ≥ min_count
# ─────────────────────────────────────────────────────────────
def extract_vocab(
        min_count: int,
        db: str, coll: str,
        filter_subreddits: Optional[List[str]] = None,
        mongo_uri: str = "mongodb://localhost:27017/"
) -> tuple[list[str], list[Tuple[str, str]]]:

    docs = return_documents(db, coll, filter_subreddits, mongo_uri=mongo_uri)

    tok_cnt   : Counter[str]         = Counter()
    bigram_cnt: Counter[Tuple[str,str]] = Counter()

    for doc in docs:
        toks = clean_and_tokenize(doc)
        tok_cnt.update(toks)
        bigram_cnt.update(zip(toks, toks[1:]))

    stopset = set(stopwords.words("english"))

    unigrams = [t for t,c in tok_cnt.items()
                if c >= min_count and t not in stopset]

    bigrams  = [(w1,w2) for (w1,w2),c in bigram_cnt.items()
                if c >= min_count and w1 not in stopset and w2 not in stopset]

    return unigrams, bigrams

# ─────────────────────────────────────────────────────────────
# 2. BERT embedding helpers
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_texts(texts: list[str], tokenizer, model,
                batch_size=64, device='cpu') -> np.ndarray:
    """Mean-pool last-layer token embeddings (excluding [CLS]/[SEP])."""
    vecs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i+batch_size]
        enc   = tokenizer(batch, return_tensors='pt',
                          padding=True, truncation=True).to(device)
        out   = model(**enc).last_hidden_state  # (B, T, H)
        # mask out [CLS] & [SEP] (first / last non-pad position)
        mask  = enc.attention_mask
        # remove CLS token (index 0)
        token_vecs = out[:,1:,:]
        token_mask = mask[:,1:]
        # mean over real tokens
        sum_vec   = (token_vecs * token_mask.unsqueeze(-1)).sum(1)
        denom     = token_mask.sum(1).unsqueeze(-1).clamp(min=1e-9)
        mean_vec  = sum_vec / denom
        vecs.append(mean_vec.cpu().numpy())
    return np.vstack(vecs)

def unit_norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-9, None)

# ─────────────────────────────────────────────────────────────
# 3. Query-time neighbour search
# ─────────────────────────────────────────────────────────────
def find_neighbors(
    query_phrase: str,
    token2vec   : dict[str, np.ndarray],
    bigrams     : list[Tuple[str,str]],
    top_k       : int,
    eps         : Optional[float] = None
):
    # embed query
    q_vec = token2vec.get(query_phrase)
    if q_vec is None:                       # phrase not a single known token
        # embed full phrase on the fly
        q_vec = embed_texts([query_phrase], tokenizer, model, device=device)[0]
    q_vec /= max(np.linalg.norm(q_vec), 1e-9)

    # score every term
    scores, phrases = [], []

    # unigrams
    for tok, vec in token2vec.items():
        sim = np.dot(vec, q_vec)
        phrases.append(tok)
        scores.append(sim)

    # bigrams
    for w1, w2 in bigrams:
        if w1 not in token2vec or w2 not in token2vec:
            continue
        sim = 0.5 * (np.dot(token2vec[w1], q_vec) +
                     np.dot(token2vec[w2], q_vec))
        phrases.append(f"{w1} {w2}")
        scores.append(sim)

    scores = np.asarray(scores)

    if eps is not None:
        idx = np.where(scores >= eps)[0]
        idx = idx[np.argsort(-scores[idx])]
        header = f"Neighbours with sim ≥ {eps}"
    else:
        idx = np.argsort(-scores)[:top_k]
        header = f"Top {top_k} neighbours"

    print(f"\n{header} for '{query_phrase}':")
    for i in idx:
        ph = phrases[i]
        sim = scores[i]
        if " " in ph:
            w1, w2 = ph.split()
            sim1 = np.dot(token2vec[w1], q_vec)
            sim2 = np.dot(token2vec[w2], q_vec)
            print(f"  {ph:<25} avg {sim:.4f} ; {w1}:{sim1:.4f} , {w2}:{sim2:.4f}")
        else:
            print(f"  {ph:<25} sim {sim:.4f}")

# ─────────────────────────────────────────────────────────────
# 4. Main CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nearest-neighbour search with BERT (unigrams + bigrams)")
    parser.add_argument("--bert_model", default="bert-base-uncased",
                        help="HF model name or local path")
    parser.add_argument("--db", default="reddit")
    parser.add_argument("--coll", default="noburp_all")
    parser.add_argument("--filter_subreddits", default=["noburp"],
                        help="comma-sep list or leave empty")
    parser.add_argument("--min_count", type=int, default=5,
                        help="min frequency for uni- & bigrams")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_npz", default=None,
                        help="Optional path to save embeddings (.npz)")
    parser.add_argument("--load_npz", default=None,
                        help="Load embeddings instead of recomputing")
    parser.add_argument("--query", required=True,
                        help="phrase to search neighbours for")
    parser.add_argument("-k", type=int, default=15,
                        help="Top-k (ignored if --eps)")
    parser.add_argument("--eps", type=float, default=None,
                        help="ε radius (cosine ≥ eps)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    logging.info("Using device: %s", device)

    # ── load / build embeddings ─────────────────────────────
    if args.load_npz:
        print(f"Loading embeddings from {args.load_npz}")
        npz = np.load(args.load_npz, allow_pickle=True)
        token_list = npz["tokens"]
        vecs       = npz["vecs"]
        token2vec  = {t: vecs[i] for i,t in enumerate(token_list)}
        bigrams    = [tuple(bg.split()) for bg in npz["bigrams"]]
        # need tokenizer/model only for query fallback
        tokenizer  = AutoTokenizer.from_pretrained(args.bert_model)
        model      = AutoModel.from_pretrained(args.bert_model).to(device)
    else:
        # mine vocab
        fsr = args.filter_subreddits if args.filter_subreddits else None
        print("Mining corpus for frequent terms …")
        unigrams, bigrams = extract_vocab(args.min_count,
                                          args.db, args.coll, fsr)
        print(f"  kept {len(unigrams)} unigrams, {len(bigrams)} bigrams")

        # embed unigrams
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        model     = AutoModel.from_pretrained(args.bert_model).to(device)
        print("Embedding unigrams with BERT …")
        vecs = embed_texts(unigrams, tokenizer, model,
                           batch_size=args.batch_size, device=device)
        vecs = unit_norm(vecs)
        token2vec = dict(zip(unigrams, vecs))

        if args.save_npz:
            np.savez_compressed(args.save_npz,
                                tokens=np.array(unigrams),
                                vecs=vecs,
                                bigrams=np.array([' '.join(b) for b in bigrams]))
            print(f"Saved embeddings to {args.save_npz}")

    # ── query time ──────────────────────────────────────────
    find_neighbors(args.query, token2vec, bigrams,
                   top_k=args.k, eps=args.eps)