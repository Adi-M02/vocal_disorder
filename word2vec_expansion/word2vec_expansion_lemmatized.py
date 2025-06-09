"""
Expand categories using Word2Vec triplet vectors and frequency filtering
with optional lemmatization and KNN, fully vectorized for speed.

usage: python word2vec_expansion/word2vec_expansion_spellcheck.py \
       --terms <path> \
       --model_dir <dir> \
       [--sim_threshold <float>] \
       [--freq_threshold <float>] \
       [--spellcheck] \
       [--lookup <lemma_lookup.json>] \
       [--topk <int>]
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
from tqdm import tqdm

sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from query_mongo import return_documents
from spellchecker_folder.spellchecker import clean_and_tokenize_spellcheck


def load_terms(path: str) -> dict[str, list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ')
                                    for term in terms]
        for category, terms in terms_map.items()
    }


def load_lookup(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_frequent_bigrams(
    min_count: int,
    tok_fn,
    lookup_map: dict[str,str],
    filter_subreddits: Optional[List[str]] = None,
    mongo_uri: str = "mongodb://localhost:27017/"
) -> List[Tuple[str, str]]:
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=filter_subreddits,
        mongo_uri=mongo_uri
    )
    counts = Counter()
    for doc in tqdm(docs, desc="Loading docs for bigrams"):
        tokens = [lookup_map.get(t, t) for t in tok_fn(doc)]
        for w1, w2 in zip(tokens, tokens[1:]):
            counts[(w1, w2)] += 1
    stops = set(stopwords.words('english'))
    return [
        bg for bg, cnt in counts.items()
        if cnt >= min_count and bg[0] not in stops and bg[1] not in stops
    ]


def embed_phrase(
    model: Word2Vec,
    phrase: str,
    tok_fn,
    lookup_map: dict[str,str]
) -> Optional[np.ndarray]:
    tokens = [lookup_map.get(t, t) for t in tok_fn(phrase)]
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def compute_triplet_vectors(
    model: Word2Vec,
    terms_map: dict[str, list[str]],
    tok_fn,
    lookup_map: dict[str,str]
) -> list[dict]:
    """
    For each category, take all 2-term combinations of its seeds.
    Embed term1, term2, and category name, then average those three vectors.
    """
    triplets = []
    for category, terms in terms_map.items():
        cat_vec = embed_phrase(model, category, tok_fn, lookup_map)
        if cat_vec is None:
            print(f"Warning: Category name '{category}' OOV; skipping")
            continue
        for term1, term2 in itertools.combinations(terms, 2):
            v1 = embed_phrase(model, term1, tok_fn, lookup_map)
            v2 = embed_phrase(model, term2, tok_fn, lookup_map)
            if v1 is None or v2 is None:
                continue
            triplets.append({
                'category': category,
                'vector': (v1 + v2 + cat_vec) / 3.0
            })
    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Expand categories using triplet + frequency filtering with lemmas and KNN"
    )
    parser.add_argument('--terms',       required=True)
    parser.add_argument('--model_dir',   required=True)
    parser.add_argument('--sim_threshold', type=float, default=0.4)
    parser.add_argument('--freq_threshold', type=float, default=0.09)
    parser.add_argument('--spellcheck',  action='store_true')
    parser.add_argument('--lookup',      default='testing/lemma_lookup.json')
    parser.add_argument('--topk',        type=int, default=None,
                        help="Use top-k KNN instead of cosine threshold")
    args = parser.parse_args()

    # pick tokenizer + lemma lookup
    TOK_FN = clean_and_tokenize_spellcheck if args.spellcheck else clean_and_tokenize
    lookup_map = load_lookup(args.lookup)
    mode = "spell-checking" if args.spellcheck else "vanilla"
    print(f"→ Using {mode} tokenizer + lemma lookup")

    # load seed terms
    terms_map = load_terms(args.terms)
    print(f"Loaded {sum(len(v) for v in terms_map.values())} seed terms")

    # extract & lemmatize bigrams
    frequent_bigrams = extract_frequent_bigrams(5, TOK_FN, lookup_map, ['noburp'])
    bigram_keep = np.array([f"{w1} {w2}" for w1, w2 in frequent_bigrams])

    # for each model
    for model_file in tqdm(['word2vec_cbow.model', 'word2vec_skipgram.model'], desc="Loading models"):
        path = os.path.join(args.model_dir, model_file)
        if not os.path.exists(path):
            parser.error(f"Model not found: {path}")
        print(f"\n>> Loading {model_file}")
        model = Word2Vec.load(path)

        # prepare unigram vocab matrix
        vocab_words  = model.wv.index_to_key
        vocab_mat    = model.wv.vectors.astype(np.float32)
        vocab_unit   = vocab_mat / np.linalg.norm(vocab_mat, axis=1, keepdims=True)

        # precompute bigram embeddings & unit vectors
        bigram_phrases = bigram_keep.tolist()
        bigram_vecs = []
        for phrase in tqdm(bigram_phrases, desc="Embedding bigrams"):
            v = embed_phrase(model, phrase, TOK_FN, lookup_map)
            if v is None:
                bigram_vecs.append(None)
            else:
                n = np.linalg.norm(v)
                bigram_vecs.append(v/n if n>0 else None)
        valid_idx = [i for i, v in enumerate(bigram_vecs) if v is not None]
        bigram_matrix = np.vstack([bigram_vecs[i] for i in valid_idx]) if valid_idx else np.zeros((0, model.vector_size), dtype=np.float32)

        # compute triplet vectors (vectorized)
        triplets = compute_triplet_vectors(model, terms_map, TOK_FN, lookup_map)
        print(f"Computed {len(triplets)} triplet vectors")

        # expand
        per_cat = {c: [] for c in terms_map}
        for tp in tqdm(triplets, desc="Expanding triplets"):
            cat, vec = tp['category'], tp['vector']
            n = np.linalg.norm(vec)
            if n == 0:
                continue
            unit = vec / n

            # unigram candidates
            if args.topk:
                uni_hits = {w for w, _ in model.wv.similar_by_vector(unit, topn=args.topk)}
            else:
                sims_u = vocab_unit @ unit
                uni_hits = {vocab_words[i] for i, v in enumerate(sims_u) if v >= args.sim_threshold}

            # bigram candidates
            if args.topk:
                sims_bg = bigram_matrix @ unit
                topk_bg = np.argsort(-sims_bg)[:args.topk]
                bg_hits = {bigram_phrases[valid_idx[i]] for i in topk_bg}
            else:
                sims_u = vocab_unit @ unit
                idx1 = np.array([model.wv.key_to_index.get(w1, -1) for w1, _ in frequent_bigrams])
                idx2 = np.array([model.wv.key_to_index.get(w2, -1) for _, w2 in frequent_bigrams])
                m1 = sims_u[idx1] >= args.sim_threshold
                m2 = sims_u[idx2] >= args.sim_threshold
                bg_hits = set(bigram_keep[m1 & m2])

            candidates = uni_hits | bg_hits
            per_cat[cat].append(candidates)

        # frequency filter & save
        for cat, subsets in tqdm(per_cat.items(), desc="Applying frequency filter"):
            q = len(subsets)
            cnt = Counter(itertools.chain.from_iterable(subsets))
            mhits = math.ceil(args.freq_threshold * q)
            ex = [t for t, c in cnt.items() if c >= mhits]
            print(f"{model_file} | {cat}: +{len(ex)} expansions (>= {mhits}/{q})")
            terms_map[cat].extend(ex)

        out_name = f"expansions_{os.path.splitext(model_file)[0]}_" \
                   f"{datetime.now().strftime('%m%d_%H%M')}.json"
        out_path = os.path.join(args.model_dir, out_name)
        with open(out_path, 'w', encoding='utf-8') as fw:
            json.dump(terms_map, fw, indent=2)
        print(f"Wrote expansions to {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()