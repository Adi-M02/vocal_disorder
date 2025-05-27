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

# allow importing your project's tokenizer
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize


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
    parser.add_argument('--terms',       required=True, help='Path to rcpd_terms.json')
    parser.add_argument('--model_dir',   required=True, help='Dir with word2vec_*.model')
    parser.add_argument('--sim_threshold', type=float, default=0.4,
                        help='Cosine‐sim cutoff (default 0.4)')
    parser.add_argument('--freq_threshold', type=float, default=0.09,
                        help='Min fraction of triplets a term must appear in (default 0.09)')
    args = parser.parse_args()

    # 1) load seed terms
    terms_map = load_terms(args.terms)
    print(f"Loaded {sum(len(v) for v in terms_map.values())} terms "
          f"across {len(terms_map)} categories")

    for model_filename in ('word2vec_cbow.model', 'word2vec_skipgram.model'):
        model_path = os.path.join(args.model_dir, model_filename)
        if not os.path.exists(model_path):
            parser.error(f"Model not found: {model_path}")
        print(f"\n>> Loading model {model_filename} …")
        model = Word2Vec.load(model_path)

        # precompute vocab embeddings + norms
        vocab_words  = model.wv.index_to_key
        vocab_matrix = model.wv.vectors
        vocab_norms  = np.linalg.norm(vocab_matrix, axis=1)

        # 2) compute triplets
        triplets = compute_triplet_vectors(model, terms_map)
        print(f"Computed {len(triplets)} triplet vectors")

        # 3) collect per‐triplet candidate sets
        per_triplet: dict[str, list[set[str]]] = {
            cat: [] for cat in terms_map
        }
        for trip in triplets:
            cat = trip['category']
            vec = trip['vector']
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            sims = vocab_matrix.dot(vec) / (vocab_norms * norm)
            idxs = np.where(sims >= args.sim_threshold)[0]
            per_triplet[cat].append({ vocab_words[i] for i in idxs })

        # 4) frequency filtering
        expansions = {}
        for cat, subsets in per_triplet.items():
            total_queries = len(subsets)
            if total_queries == 0:
                expansions[cat] = []
                continue
            # Minimum times a term must appear
            min_count = math.ceil(args.freq_threshold * total_queries)
            # Count across all triplet‐sets
            counter = Counter()
            for s in subsets:
                counter.update(s)
            # Keep those meeting both sim + freq criteria
            kept = [term for term, cnt in counter.items() if cnt >= min_count]
            expansions[cat] = kept
            print(f"{model_filename} | {cat}: "
                  f"{len(terms_map[cat])} seeds → +{len(kept)} expansions "
                  f"(freq ≥ {min_count}/{total_queries})")

        # 5) merge seeds + expansions & write JSON
        merged = {
            cat: terms_map[cat] + expansions[cat]
            for cat in terms_map
        }
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        out_file = os.path.join(
            args.model_dir,
            f"expansions_{model_filename.replace('.model','')}_{timestamp}.json"
        )
        with open(out_file, 'w', encoding='utf-8') as fw:
            json.dump(merged, fw, indent=2)
        print(f"Wrote expansions to {out_file}")

    print("\nDone.")


if __name__ == '__main__':
    main()
