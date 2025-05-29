#!/usr/bin/env python3
import sys
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
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


def reduce_PCA(df: pd.DataFrame, pca_components: int = 2) -> pd.DataFrame:
    """
    Reduce the 'dim*' columns of df down to 2 or 3 PCA components.

    Prints:
      - Explained variance ratio (sum of the top components) and the lost variance
      - Reconstruction MSE

    Returns a new DataFrame with 'x','y' (and 'z' if pca_components==3) columns added.
    """
    if pca_components not in (2, 3):
        raise ValueError(f"pca_components must be 2 or 3, got {pca_components}")

    # 1) Pull out your high-dimensional data
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    # 2) Fit PCA
    pca = PCA(n_components=pca_components, random_state=42)
    coords = pca.fit_transform(X)

    # 3) Print variance capture
    explained = float(pca.explained_variance_ratio_.sum())
    lost = 1.0 - explained
    print(f"PCA: {pca_components} components explain {explained:.2%} of the variance, losing {lost:.2%}")

    # 4) Print reconstruction MSE
    X_recon = pca.inverse_transform(coords)
    mse = np.mean((X - X_recon) ** 2)
    print(f"PCA reconstruction MSE: {mse:.6f}")

    # 5) Attach coords (x,y, and z if requested)
    out = df.copy()
    out['x'] = coords[:, 0]
    out['y'] = coords[:, 1]
    if pca_components == 3:
        out['z'] = coords[:, 2]
    return out


def reduce_tsne(df: pd.DataFrame, n_components: int = 2, **tsne_kwargs) -> pd.DataFrame:
    """
    Runs t-SNE on the 'dim*' columns of df, producing 2D or 3D coordinates.
    Prints KL divergence and an embedding‐variance ratio so you can compare how
    much of the original spread is captured.
    """
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    tsne = TSNE(n_components=n_components, random_state=42, **tsne_kwargs)
    coords = tsne.fit_transform(X)

    # Print error metrics
    if hasattr(tsne, 'kl_divergence_'):
        print(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")
    # variance‐ratio proxy: sum var(embedding)/sum var(original)
    orig_var = np.var(X, axis=0).sum()
    emb_var  = np.var(coords, axis=0).sum()
    ratio    = emb_var / orig_var
    print(f"t-SNE embedding variance ratio: {ratio:.2%}")

    # Attach coords
    out = df.copy()
    out['x'] = coords[:, 0]
    out['y'] = coords[:, 1]
    if n_components == 3:
        out['z'] = coords[:, 2]
    return out


def reduce_umap(df: pd.DataFrame, n_components: int = 2, **umap_kwargs) -> pd.DataFrame:
    """
    Runs UMAP on the 'dim*' columns of df, producing 2D or 3D coordinates.
    Prints a variance‐ratio proxy so you can see how it compares to PCA and t-SNE.
    """
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    um = umap.UMAP(n_components=n_components, random_state=42, **umap_kwargs)
    coords = um.fit_transform(X)

    # variance‐ratio proxy
    orig_var = np.var(X, axis=0).sum()
    emb_var  = np.var(coords, axis=0).sum()
    ratio    = emb_var / orig_var
    print(f"UMAP embedding variance ratio: {ratio:.2%}")

    out = df.copy()
    out['x'] = coords[:, 0]
    out['y'] = coords[:, 1]
    if n_components == 3:
        out['z'] = coords[:, 2]
    return out


def plot_and_save(df: pd.DataFrame, title: str, out_html: str):
    dims = ['x', 'y', 'z']
    has = [d for d in dims if d in df.columns]

    if 'z' in has:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='category',
            hover_data=['term'],
            title=title,
            width=800, height=600
        )
    else:
        fig = px.scatter(
            df, x='x', y='y',
            color='category',
            hover_data=['term'],
            title=title,
            width=800, height=600
        )

    fig.write_html(out_html)
    print(f" → saved {out_html}")


if __name__ == "__main__":
    # === Configuration ===
    TERMS_PATH     = 'rcpd_terms.json'
    MODEL_DIR      = "word2vec_expansion/word2vec_05_26_16_04"
    # Timestamp for output folder
    ts             = datetime.now().strftime("%m_%d_%H_%M")
    viz_dir        = os.path.join(MODEL_DIR, f"embedding_visualization_{ts}")
    SIM_THRESHOLD  = 0.4    # unused here but available for future filtering
    FREQ_THRESHOLD = 0.09   # unused here but available for future filtering

    # Ensure necessary files/folders exist
    cbow_path = os.path.join(MODEL_DIR, "word2vec_cbow.model")
    skip_path = os.path.join(MODEL_DIR, "word2vec_skipgram.model")
    for path in (TERMS_PATH, cbow_path, skip_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Create output directory
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Writing visualization HTML into: {viz_dir}\n")

    # Load terms and give feedback
    terms_map   = load_terms(TERMS_PATH)
    total_terms = sum(len(v) for v in terms_map.values())
    print(f"Loaded {total_terms} terms across {len(terms_map)} categories\n")

    # Process each model
    for model_path, label in [(cbow_path, "CBOW"), (skip_path, "SkipGram")]:
        print(f"[{label}] Loading model from {model_path} …")
        model = Word2Vec.load(model_path)

        print(f"[{label}] Extracting embeddings with mean-pooling …")
        df = extract_embeddings(model, terms_map)
        print(f"  → Embedded {len(df)} / {total_terms} terms")

        print(f"[{label}] Reducing to 2D via PCA …")
        df2 = reduce_PCA(df, pca_components=3)

        out_html = os.path.join(viz_dir, f"embeddings_{label.lower()}.html")
        print(f"[{label}] Building interactive plot …")
        plot_and_save(df2, f"{label} embeddings (2D PCA)", out_html)
        print()

    print("Done.")
