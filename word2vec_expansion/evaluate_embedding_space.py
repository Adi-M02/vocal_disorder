import sys
import os
import json
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize


def load_terms(path: str) -> dict[str, list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }


def extract_embeddings(model: Word2Vec, terms_map: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for category, terms in terms_map.items():
        for term in terms:
            tokens = clean_and_tokenize(term)
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            if not vecs:
                continue
            mean_vec = np.mean(vecs, axis=0)
            row = {'term': term, 'category': category}
            for i, val in enumerate(mean_vec):
                row[f'dim{i}'] = float(val)
            rows.append(row)
        # also embed category label itself
        tokens = clean_and_tokenize(category)
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            row = {'term': category, 'category': category}
            for i, val in enumerate(mean_vec):
                row[f'dim{i}'] = float(val)
            rows.append(row)
    return pd.DataFrame(rows)


def find_vocab_neighbors(model: Word2Vec, model_label: str, query_term: str, k: int):
    tokens = clean_and_tokenize(query_term)
    if len(tokens) != 1:
        print(f"[{model_label}] Please query a single unigram, got: {tokens}")
        return
    token = tokens[0]
    if token not in model.wv:
        print(f"[{model_label}] '{token}' not found in the model vocabulary.")
        return
    neighbors = model.wv.most_similar(positive=[token], topn=k)
    print(f"[{model_label}] Top {k} neighbors for '{token}':")
    for neigh, sim in neighbors:
        print(f"  {neigh} (similarity: {sim:.4f})")

# ————————————————————————————————————————————————
# Dimensionality reduction and plotting functions remain unchanged
# (reduce_PCA, reduce_tsne, reduce_umap, plot_and_save)
# Insert your existing routines here.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models and query nearest neighbors in full vocab.")
    parser.add_argument('-d', '--model_dir', type=str, default='word2vec_expansion/word2vec_05_26_16_04', help='Directory containing Word2Vec models')
    parser.add_argument('--query', type=str, help='Single unigram to query nearest neighbors')
    parser.add_argument('-k', type=int, default=5, help='Number of nearest neighbors for vocab query')
    args = parser.parse_args()

    MODEL_DIR  = args.model_dir

    for model_file, label in [
        (os.path.join(MODEL_DIR, "word2vec_cbow.model"), "CBOW"),
        (os.path.join(MODEL_DIR, "word2vec_skipgram.model"), "SkipGram")
    ]:
        print(f"[{label}] Loading model from {model_file}...")
        model = Word2Vec.load(model_file)

        # Optionally query full vocabulary
        if args.query:
            find_vocab_neighbors(model, label, args.query, args.k)

        # Visualization steps would go here (unchanged)

    print("Done.")
