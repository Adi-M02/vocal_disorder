import sys
import os
import json
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import umap

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
        # also embed category as a term
        tokens = clean_and_tokenize(category)
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            row = {'term': category, 'category': category}
            for i, val in enumerate(mean_vec):
                row[f'dim{i}'] = float(val)
            rows.append(row)
    return pd.DataFrame(rows)


def build_nn_index(df: pd.DataFrame) -> tuple[NearestNeighbors, np.ndarray]:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values
    nn = NearestNeighbors(metric='cosine').fit(X)
    return nn, X


def find_neighbors(model_label: str, df: pd.DataFrame, nn: NearestNeighbors, vector_matrix: np.ndarray, query_term: str, k: int):
    tokens = clean_and_tokenize(query_term)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        print(f"[{model_label}] No in-vocab tokens for '{query_term}'")
        return
    mean_vec = np.mean(vecs, axis=0).reshape(1, -1)
    dists, idxs = nn.kneighbors(mean_vec, n_neighbors=k+1)
    print(f"[{model_label}] Nearest neighbors to '{query_term}':")
    count = 0
    for dist, idx in zip(dists[0], idxs[0]):
        term = df.iloc[idx]['term']
        if term.lower() == query_term.lower():
            continue
        print(f"  {term} (cosine distance: {dist:.4f})")
        count += 1
        if count >= k:
            break

# ————————————————————————————————————————————————
# Dimensionality reduction routines (unchanged)
# [reduce_PCA, reduce_tsne, reduce_umap, plot_and_save as before]
# ...

# For brevity, include your existing reduce_* and plot_and_save functions here without change
# (omitted in this snippet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embeddings and/or find nearest neighbors")
    parser.add_argument('-t', '--terms', type=str, default='rcpd_terms.json', help='Path to term map JSON')
    parser.add_argument('-d', '--model_dir', type=str, default='word2vec_expansion/word2vec_05_26_16_04', help='Directory containing Word2Vec models')
    parser.add_argument('--query', type=str, help='Term to query nearest neighbors for')
    parser.add_argument('-k', type=int, default=5, help='Number of nearest neighbors')
    args = parser.parse_args()

    TERMS_PATH = args.terms
    MODEL_DIR  = args.model_dir
    ts         = datetime.now().strftime("%m_%d_%H_%M")
    viz_dir    = os.path.join(MODEL_DIR, f"embedding_visualization_{ts}")
    os.makedirs(viz_dir, exist_ok=True)

    metrics_path = os.path.join(viz_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        mf.write(f"Embedding reduction metrics — run at {datetime.now()}\n\n")

    terms_map = load_terms(TERMS_PATH)
    print(f"Loaded {sum(len(v) for v in terms_map.values())} terms across {len(terms_map)} categories")

    for model_file, label in [
        (os.path.join(MODEL_DIR, "word2vec_cbow.model"), "CBOW"),
        (os.path.join(MODEL_DIR, "word2vec_skipgram.model"), "SkipGram")
    ]:
        print(f"[{label}] Loading model...")
        model = Word2Vec.load(model_file)
        df    = extract_embeddings(model, terms_map)

        # Build nearest-neighbors index
        nn, X = build_nn_index(df)
        if args.query:
            find_neighbors(label, df, nn, X, args.query, args.k)

        # Visualization steps (PCA, t-SNE, UMAP) as before
        # ... your existing loops here ...

    print("Done.")
