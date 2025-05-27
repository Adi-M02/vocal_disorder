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
    # keys are category names
    return {category.replace('_', ' '): terms for category, terms in terms_map.items()}


def extract_term_and_query_embeddings(
    model: Word2Vec,
    terms_map: dict[str, list[str]],
    queries: list[str]
) -> pd.DataFrame:
    rows = []
    # 1) embed every seed term in its category
    for category, terms in terms_map.items():
        for term in terms:
            tokens = clean_and_tokenize(term)
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            if not vecs:
                continue
            mean_vec = np.mean(vecs, axis=0)
            row = {'term': term, 'category': category, 'type': 'seed'}
            for i, val in enumerate(mean_vec):
                row[f'dim{i}'] = float(val)
            rows.append(row)

    # 2) embed your additional query words/phrases
    for q in queries:
        tokens = clean_and_tokenize(q)
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            continue
        mean_vec = np.mean(vecs, axis=0)
        row = {'term': q, 'category': 'Additional', 'type': 'query'}
        for i, val in enumerate(mean_vec):
            row[f'dim{i}'] = float(val)
        rows.append(row)

    return pd.DataFrame(rows)


def reduce_to_2d(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PCA to reduce dimension columns to x,y."""
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[dim_cols].values)
    df['x'], df['y'] = coords[:, 0], coords[:, 1]
    return df


def plot_and_save(df: pd.DataFrame, title: str, out_html: str):
    """Scatter plot colored by category and symbolized by type."""
    fig = px.scatter(
        df,
        x='x', y='y',
        color='category',
        symbol='type',
        hover_data=['term'],
        title=title,
        width=800, height=600
    )
    fig.write_html(out_html)
    print(f" → saved {out_html}")


if __name__ == "__main__":
    # === Configuration ===
    TERMS_PATH      = "rcpd_terms.json"
    MODEL_DIR       = "word2vec_expansion/word2vec_05_26_16_04"
    ADDITIONAL_WORDS = [
        "still", "second"
    ]

    # Prepare output directory
    ts = datetime.now().strftime("%m_%d_%H_%M")
    viz_dir = os.path.join(MODEL_DIR, f"term_embedding_viz_{ts}")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Writing visualizations into: {viz_dir}\n")

    # Validate inputs
    cbow_path = os.path.join(MODEL_DIR, "word2vec_cbow.model")
    skip_path = os.path.join(MODEL_DIR, "word2vec_skipgram.model")
    for path in (TERMS_PATH, cbow_path, skip_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load categories
    terms_map = load_terms(TERMS_PATH)
    categories = list(terms_map.keys())
    print(f"Loaded {len(categories)} categories\n")

    # Iterate models
    for model_path, label in [(cbow_path, "CBOW"), (skip_path, "SkipGram")]:
        print(f"[{label}] Loading model from {model_path} …")
        model = Word2Vec.load(model_path)

        print(f"[{label}] Embedding {sum(len(v) for v in terms_map.values())} seed terms + {len(ADDITIONAL_WORDS)} queries …")
        df = extract_term_and_query_embeddings(model, terms_map, ADDITIONAL_WORDS)
        print(f"  → Embedded {len(df)} items")

        print(f"[{label}] Reducing to 2D via PCA …")
        df2 = reduce_to_2d(df)

        out_html = os.path.join(viz_dir, f"cat_query_{label.lower()}.html")
        print(f"[{label}] Building interactive plot …")
        plot_and_save(df2, f"{label} Category + Query Embeddings", out_html)
        print()

    print("Done.")
