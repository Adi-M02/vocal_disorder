#!/usr/bin/env python3
import os
import json
import argparse
import sys
import itertools
import numpy as np
from gensim.models import Word2Vec
from datetime import datetime

# allow importing your project's tokenizer and cleaning function
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize


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


def embed_phrase(model: Word2Vec, phrase: str) -> np.ndarray | None:
    """
    Tokenize `phrase`, lookup each token in model.wv, then mean-pool.
    Returns None if no tokens are in-vocab.
    """
    tokens = clean_and_tokenize(phrase)
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def compute_triplet_vectors(
    model: Word2Vec,
    terms_map: dict[str, list[str]]
) -> list[dict]:
    """
    For each category, take all 2-term combinations of its terms.
    Embed term1, term2, and category name, then average those three vectors.
    Returns a list of {'category', 'term1', 'term2', 'vector'}.
    """
    triplets = []
    for category, terms in terms_map.items():
        cat_vec = embed_phrase(model, category)
        if cat_vec is None:
            print(f"Warning: Category name '{category}' is OOV, skipping category")
            continue
        for term1, term2 in itertools.combinations(terms, 2):
            v1 = embed_phrase(model, term1)
            v2 = embed_phrase(model, term2)
            if v1 is None or v2 is None:
                continue
            mean_vec = (v1 + v2 + cat_vec) / 3.0
            triplets.append({
                'category': category,
                'term1': term1,
                'term2': term2,
                'vector': mean_vec
            })
    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Expand categories using triplet-centroid similarity from Word2Vec models"
    )
    parser.add_argument(
        '--terms', required=True,
        help='Path to your rcpd_terms.json'
    )
    parser.add_argument(
        '--model_dir', required=True,
        help='Directory containing word2vec_cbow.model and word2vec_skipgram.model'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.4,
        help='Cosine similarity cutoff for expansion'
    )
    args = parser.parse_args()

    # load seed terms
    terms_map = load_terms(args.terms)
    print(f"Loaded {sum(len(v) for v in terms_map.values())} terms across {len(terms_map)} categories")

    for model_filename in ('word2vec_cbow.model', 'word2vec_skipgram.model'):
        model_path = os.path.join(args.model_dir, model_filename)
        if not os.path.exists(model_path):
            parser.error(f"Model not found: {model_path}")
        print(f"\n>> Loading model {model_filename}...")
        model = Word2Vec.load(model_path)

        # precompute vocabulary embeddings and norms
        vocab_words = model.wv.index_to_key
        vocab_matrix = model.wv.vectors
        vocab_norms = np.linalg.norm(vocab_matrix, axis=1)

        # compute triplet vectors
        triplets = compute_triplet_vectors(model, terms_map)
        print(f"Computed {len(triplets)} triplet vectors")

        # accumulate expansions per category
        expansions = {cat: set() for cat in terms_map}
        for trip in triplets:
            vec = trip['vector']
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            sims = vocab_matrix.dot(vec) / (vocab_norms * norm)
            idxs = np.where(sims > args.threshold)[0]
            for i in idxs:
                expansions[trip['category']].add(vocab_words[i])

        # merge original seeds and expansions
        merged = {}
        for cat, orig in terms_map.items():
            extra = [w for w in sorted(expansions[cat]) if w not in orig]
            merged[cat] = orig + extra
            print(f"{model_filename} | {cat}: {len(orig)} → {len(merged[cat])}")

        # write out JSON
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        out_file = os.path.join(
            args.model_dir,
            f"expansions_{model_filename.replace('.model','')}_{timestamp}.json"
        )
        with open(out_file, 'w', encoding='utf-8') as fw:
            json.dump(merged, fw, indent=2)
        print(f"Wrote expansions to {out_file}")

if __name__ == '__main__':
    main()
