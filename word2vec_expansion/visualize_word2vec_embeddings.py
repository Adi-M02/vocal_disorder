#!/usr/bin/env python3
import json
import os
import argparse
import datetime

from gensim.models import Word2Vec
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

def load_terms(path):
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {category.replace('_', ' '): [term.replace('_', ' ') for term in terms] for category, terms in terms_map.items()}

def extract_embeddings(model: Word2Vec, terms_map: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for category, terms in terms_map.items():
        for term in terms:
            if term in model.wv:
                vec = model.wv[term]
                rows.append({
                    'term': term,
                    'category': category,
                    **{f'dim{i}': float(vec[i]) for i in range(len(vec))}
                })
    return pd.DataFrame(rows)

def reduce_to_2d(df: pd.DataFrame) -> pd.DataFrame:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[dim_cols].values)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
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

def main():
    p = argparse.ArgumentParser(
        description="Visualize CBOW & Skip-gram embeddings from a model folder"
    )
    p.add_argument(
        '--terms', required=True,
        help="Path to your rcpd_terms.json"
    )
    p.add_argument(
        '--model_dir', required=True,
        help="Folder containing word2vec_cbow.model & word2vec_skipgram.model"
    )
    args = p.parse_args()

    # verify input
    cbow_path     = os.path.join(args.model_dir, "word2vec_cbow.model")
    skip_path     = os.path.join(args.model_dir, "word2vec_skipgram.model")
    for pth in (cbow_path, skip_path, args.terms):
        if not os.path.exists(pth):
            p.error(f"Not found: {pth}")

    # prepare output folder
    ts = datetime.datetime.now().strftime("%m_%d_%H_%M")
    viz_dir = os.path.join(args.model_dir, f"embedding_visualization_{ts}")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Writing visualization HTML into: {viz_dir}\n")

    # load terms
    terms_map = load_terms(args.terms)
    total_terms = sum(len(v) for v in terms_map.values())
    print(f"Loaded {total_terms} terms across {len(terms_map)} categories\n")

    # process each model
    for model_path, label in [(cbow_path, "CBOW"), (skip_path, "SkipGram")]:
        print(f"[{label}] Loading model from {model_path} …")
        model = Word2Vec.load(model_path)

        print(f"[{label}] Extracting in-vocab embeddings …")
        df = extract_embeddings(model, terms_map)
        print(f"  → Found {len(df)} / {total_terms} terms in vocabulary")

        print(f"[{label}] Reducing to 2D via PCA …")
        df2 = reduce_to_2d(df)

        out_html = os.path.join(viz_dir, f"embeddings_{label.lower()}.html")
        print(f"[{label}] Building interactive plot …")
        plot_and_save(df2, f"{label} embeddings (2D PCA)", out_html)
        print()

    print("Done.")

if __name__ == "__main__":
    main()
