#!/usr/bin/env python3
"""
Generate interactive 2D embedding visualizations with comprehensive quality metrics.

Usage:
    python visualize_embeddings.py \
        --model_path /path/to/model.model \
        --json_terms terms.json \
        --txt_terms terms.txt
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, trustworthiness
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
import umap.umap_ as umap
import plotly.express as px
import warnings

# Suppress specific UMAP warnings
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value.*overridden to.*random_state.*"
)

# Custom tokenizer imports
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list

def load_lookup(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

lookup = load_lookup("testing/lemma_lookup.json")

def tok_fn(text: str) -> List[str]:
    tokens = clean_and_tokenize(text)
    tokens = [lookup.get(t, t) for t in tokens]
    tokens = spellcheck_token_list(tokens)
    tokens = [lookup.get(t, t) for t in tokens]
    return tokens

# Helper functions for neighborhood metrics

def compute_neighbor_ranks(dists: np.ndarray) -> List[List[int]]:
    n = dists.shape[0]
    ranks = []
    for i in range(n):
        idx = np.argsort(dists[i])
        idx = idx[idx != i]
        ranks.append(idx.tolist())
    return ranks


def compute_continuity(high_ranks: List[List[int]], low_ranks: List[List[int]], k: int) -> float:
    n = len(high_ranks)
    total = 0.0
    for i in range(n):
        U = set(high_ranks[i][:k]) - set(low_ranks[i][:k])
        for j in U:
            rank_low = low_ranks[i].index(j) + 1
            total += (rank_low - k)
    denom = n * k * (2*n - 3*k - 1)
    return 1 - (2 * total / denom) if denom else 0.0


def compute_jaccard(high_ranks: List[List[int]], low_ranks: List[List[int]], k: int) -> float:
    n = len(high_ranks)
    total = 0.0
    for i in range(n):
        set_h = set(high_ranks[i][:k])
        set_l = set(low_ranks[i][:k])
        union = set_h | set_l
        if not union:
            continue
        total += len(set_h & set_l) / len(union)
    return total / n if n else 0.0


def load_terms_json(path: str) -> List[str]:
    """
    Load terms from a JSON file where values are lists of terms.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        raw = []
        for vals in data.values():
            if isinstance(vals, list):
                raw.extend(vals)
            else:
                raw.append(vals)
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError('JSON must contain a dict of lists or a list of terms')
    tokens: List[str] = []
    for term in raw:
        tokens.extend(tok_fn(term))
    seen, unique = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def load_terms_txt(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = [t.strip() for t in f.read().split(',') if t.strip()]
    tokens: List[str] = []
    for term in raw:
        tokens.extend(tok_fn(term))
    seen, unique = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def filter_terms(terms: List[str], model: Word2Vec) -> List[str]:
    vocab = set(model.wv.key_to_index.keys())
    return [t for t in terms if t in vocab]


def extract_embeddings(terms: List[str], model: Word2Vec) -> np.ndarray:
    return np.array([model.wv[t] for t in terms])


def plot_embeddings(X: np.ndarray, terms: List[str], sources: List[str], method: str, outdir: str):
    df = pd.DataFrame({'term': terms, 'x': X[:,0], 'y': X[:,1], 'source': sources})
    fig = px.scatter(df, x='x', y='y', color='source', hover_name='term', title=f'{method.upper()}')
    path = os.path.join(outdir, f'{method}.html')
    fig.write_html(path)
    print(f'Saved {method} to {path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings with metrics')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--json_terms', required=True)
    parser.add_argument('--txt_terms', required=True)
    args = parser.parse_args()

    model = Word2Vec.load(args.model_path)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    base = os.path.dirname(args.model_path) or '.'
    timestamp = datetime.now().strftime('%m_%d_%H_%M')
    outdir = os.path.join(base, f'{model_name}_{timestamp}')
    os.makedirs(outdir, exist_ok=True)

    terms_j = filter_terms(load_terms_json(args.json_terms), model)
    terms_t = filter_terms(load_terms_txt(args.txt_terms), model)

    # merge, dedupe, preferring JSON label on overlap
    seen, terms, sources = set(), [], []
    for term, src in zip(terms_j + terms_t, ['json']*len(terms_j) + ['txt']*len(terms_t)):
        if term not in seen:
            seen.add(term)
            terms.append(term)
            sources.append(src)
    embeddings = extract_embeddings(terms, model)

    # compute high-D metrics
    high_dist_vec = pdist(embeddings)
    high_mat = pairwise_distances(embeddings)
    high_ranks = compute_neighbor_ranks(high_mat)

    metrics: Dict[str, dict] = {}
    methods = {
        'pca': PCA(n_components=2),
        'tsne': TSNE(n_components=2, random_state=42),
        'umap': umap.UMAP(n_components=2, random_state=42),
        'mds': MDS(n_components=2, random_state=42)
    }

    for name, dr in methods.items():
        X = dr.fit_transform(embeddings)
        met = {}
        if name == 'pca':
            met['explained_variance_ratio'] = dr.explained_variance_ratio_.tolist()
            met['cumulative_explained_variance'] = float(sum(dr.explained_variance_ratio_))
        elif name == 'tsne':
            met['kl_divergence'] = float(dr.kl_divergence_)
        elif name == 'umap':
            recon = getattr(dr, 'reconstruction_error_', None)
            met['reconstruction_error'] = float(recon) if recon is not None else None
        else:
            met['stress'] = float(dr.stress_)

        low_dist_vec = pdist(X)
        spearman_corr, _ = spearmanr(high_dist_vec, low_dist_vec)
        pearson_corr, _ = pearsonr(high_dist_vec, low_dist_vec)
        met['spearman'] = float(spearman_corr)
        met['pearson_r'] = float(pearson_corr)
        met['pearson_r2'] = float(pearson_corr**2)

        ks = [k for k in (5,10,20) if k < embeddings.shape[0]]
        low_mat = pairwise_distances(X)
        low_ranks = compute_neighbor_ranks(low_mat)
        met['trustworthiness'] = {k: trustworthiness(embeddings, X, n_neighbors=k) for k in ks}
        met['continuity'] = {k: compute_continuity(high_ranks, low_ranks, k) for k in ks}
        met['jaccard'] = {k: compute_jaccard(high_ranks, low_ranks, k) for k in ks}

        metrics[name] = met
        plot_embeddings(X, terms, sources, name, outdir)

    with open(os.path.join(outdir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metrics.json in {outdir}')

    print("\nHyperparameter suggestions:")
    print("- PCA: n_components, whiten, svd_solver")
    print("- t-SNE: perplexity, learning_rate, n_iter, init")
    print("- UMAP: n_neighbors, min_dist, metric, n_epochs")
    print("- MDS: n_init, max_iter, dissimilarity")

if __name__ == '__main__':
    main()
