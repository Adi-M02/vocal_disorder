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

# adjust path as needed so we can import your tokenizer
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


def reduce_to_2d(df: pd.DataFrame) -> pd.DataFrame:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    # 1) Explained-variance “error”
    explained = float(pca.explained_variance_ratio_.sum())
    lost = 1.0 - explained
    print(f"PCA: 2 components explain {explained:.2%} of the variance, losing {lost:.2%}")

    # 2) Reconstruction MSE
    X_recon = pca.inverse_transform(coords)
    mse = np.mean((X - X_recon) ** 2)
    print(f"PCA reconstruction MSE: {mse:.6f}")

    # attach the 2D coords back to your dataframe
    df = df.copy()
    df['x'], df['y'] = coords[:, 0], coords[:, 1]
    return df


def plot_and_save(df: pd.DataFrame, title: str, out_html: str):
    fig = px.scatter(
        df,
        x='x', y='y',
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
        df2 = reduce_to_2d(df)

        out_html = os.path.join(viz_dir, f"embeddings_{label.lower()}.html")
        print(f"[{label}] Building interactive plot …")
        plot_and_save(df2, f"{label} embeddings (2D PCA)", out_html)
        print()

    print("Done.")
