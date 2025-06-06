"""
Expand categories using Word2Vec triplet vectors and frequency filtering.
usage: python word2vec_expansion/word2vec_expansion_spellcheck.py --terms <path> --model_dir <dir> --sim_threshold <float> --freq_threshold <float> --spellcheck
"""
import os
import sys
import json
import math
import argparse
import itertools
import numpy as np
from gensim.models import Word2Vec
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Optional
from nltk.corpus import stopwords

# allow importing your project's tokenizer and spellchecker
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from query_mongo import return_documents
from spellchecker_folder.spellchecker import clean_and_tokenize_spellcheck


# helper function to extract frequent bigrams
def extract_frequent_bigrams(
    min_count: int,
    db_name: str,
    collection_name: str,
    filter_subreddits: Optional[List[str]] = None,
    mongo_uri: str = "mongodb://localhost:27017/"
) -> List[Tuple[str, str]]:

    docs = return_documents(
        db_name,
        collection_name,
        filter_subreddits,
        mongo_uri=mongo_uri
    )

    bigram_counts: Counter[Tuple[str, str]] = Counter()

    for doc in docs:
        tokens = TOK_FN(doc)
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[(w1, w2)] += 1

    stop_words = set(stopwords.words('english'))
    frequent = [
        bigram
        for bigram, cnt in bigram_counts.items()
        if cnt >= min_count and (bigram[0] not in stop_words and bigram[1] not in stop_words)
    ]
    return frequent


def load_terms(path: str) -> dict[str, list[str]]:
    """Load category→list-of-terms from JSON, replacing underscores with spaces."""
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }


def embed_phrase(model: Word2Vec, phrase: str) -> np.ndarray | None:
    """Tokenize `phrase`, lookup each token in model.wv, then mean-pool."""
    tokens = TOK_FN(phrase)
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def compute_triplet_vectors(
    model: Word2Vec,
    terms_map: dict[str, list[str]]
) -> list[dict]:
    """
    For each category, take all 2-term combinations of its seeds.
    Embed term1, term2, and category name, then average those three vectors.
    """
    triplets = []
    for category, terms in terms_map.items():
        cat_vec = embed_phrase(model, category)
        if cat_vec is None:
            print(f"Warning: Category name '{category}' OOV; skipping")
            continue
        for term1, term2 in itertools.combinations(terms, 2):
            v1 = embed_phrase(model, term1)
            v2 = embed_phrase(model, term2)
            if v1 is None or v2 is None:
                continue
            triplets.append({
                'category': category,
                'vector': (v1 + v2 + cat_vec) / 3.0
            })
    return triplets

def main():
    parser = argparse.ArgumentParser(
        description="Expand categories using triplet + frequency filtering"
    )
    parser.add_argument('--terms',  required=True, help='Path to rcpd_terms.json')
    parser.add_argument('--model_dir', required=True, help='Dir with word2vec_*.model')
    parser.add_argument('--sim_threshold', type=float, default=0.4)
    parser.add_argument('--freq_threshold', type=float, default=0.09)
    parser.add_argument('--spellcheck', action='store_true')
    args = parser.parse_args()

    # ── choose tokenizer ───────────────────────────────────────────────
    global TOK_FN
    TOK_FN = clean_and_tokenize_spellcheck if args.spellcheck else clean_and_tokenize
    print("→ Using", "spell-checking" if args.spellcheck else "vanilla", "tokenizer")

    # ── seed terms ─────────────────────────────────────────────────────
    terms_map = load_terms(args.terms)
    print(f"Loaded {sum(len(v) for v in terms_map.values())} seed terms")

    # ── frequent bigrams (once) ────────────────────────────────────────
    frequent_bigrams = extract_frequent_bigrams(
        min_count=5,
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"]
    )
    bigram_phrases = [' '.join(bg) for bg in frequent_bigrams]
    print(f"Extracted {len(frequent_bigrams)} frequent bigrams")

    # ── iterate over models ────────────────────────────────────────────
    for model_filename in ('word2vec_cbow.model', 'word2vec_skipgram.model'):
        path = os.path.join(args.model_dir, model_filename)
        if not os.path.exists(path):
            parser.error(f"Model not found: {path}")
        print(f"\n>> Loading {model_filename} …")
        model = Word2Vec.load(path)

        # Vocab & unit vectors
        vocab_words   = model.wv.index_to_key
        vocab_matrix  = model.wv.vectors.astype(np.float32)
        vocab_norms   = np.linalg.norm(vocab_matrix, axis=1, keepdims=True)
        vocab_unit    = vocab_matrix / np.clip(vocab_norms, 1e-9, None)

        # Map token → row index
        tok2idx = {tok: i for i, tok in enumerate(vocab_words)}

        # Build arrays of token-indices for bigrams (skip OOV pairs)
        idx1, idx2, bigram_keep = [], [], []
        for (w1, w2), phrase in zip(frequent_bigrams, bigram_phrases):
            if w1 in tok2idx and w2 in tok2idx:
                idx1.append(tok2idx[w1])
                idx2.append(tok2idx[w2])
                bigram_keep.append(phrase)
        idx1, idx2 = np.asarray(idx1), np.asarray(idx2)
        bigram_keep = np.asarray(bigram_keep)
        print(f"  {len(bigram_keep)} bigrams retained after OOV filtering")

        # Prototype (triplet) vectors
        triplets = compute_triplet_vectors(model, terms_map)
        print(f"Computed {len(triplets)} triplet vectors")

        per_triplet: dict[str, list[set[str]]] = {c: [] for c in terms_map}

        for tp in triplets:
            cat, vec = tp['category'], tp['vector']
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            vec_unit = vec / vec_norm

            # ---- unigrams (vectorised) -------------------------------------
            sims_u = vocab_unit @ vec_unit
            uni_hits = sims_u >= args.sim_threshold
            candidates = {vocab_words[i] for i in np.where(uni_hits)[0]}

            # ---- bigrams (vectorised, BOTH tokens must hit) ----------------
            sims1 = sims_u[idx1]          # cosine for first token in each bigram
            sims2 = sims_u[idx2]          # cosine for second token
            mask  = (sims1 >= args.sim_threshold) & (sims2 >= args.sim_threshold)
            candidates.update(bigram_keep[mask])

            per_triplet[cat].append(candidates)

        # ── frequency filtering per category ───────────────────────────
        expansions = {}
        for cat, subsets in per_triplet.items():
            q = len(subsets)
            if q == 0:
                expansions[cat] = []
                continue
            min_hits = math.ceil(args.freq_threshold * q)
            counter  = Counter()
            for s in subsets:
                counter.update(s)
            expansions[cat] = [t for t, c in counter.items() if c >= min_hits]
            print(f"{model_filename} | {cat}: "
                  f"{len(terms_map[cat])} seeds → +{len(expansions[cat])} expansions "
                  f"(freq ≥ {min_hits}/{q})")

        # ── merge & save ------------------------------------------------
        merged = {c: terms_map[c] + expansions[c] for c in terms_map}
        model_type = "cbow" if "cbow" in model_filename else "skipgram"
        out_path = os.path.join(
            args.model_dir,
            f"expansions_{model_type}_{datetime.now().strftime('%m_%d_%H_%M')}.json"
        )
        with open(out_path, 'w', encoding='utf-8') as fw:
            json.dump(merged, fw, indent=2)
        print(f"Wrote expansions to {out_path}")

    print("\nDone.")

if __name__ == '__main__':
    main() 