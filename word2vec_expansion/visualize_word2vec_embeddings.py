import sys
import os
import json
import argparse
import datetime

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
        category.replace('_', ' '): [term.replace('_', ' ' ) for term in terms]
        for category, terms in terms_map.items()
    }


def extract_embeddings(model: Word2Vec, terms_map: dict[str, list[str]]) -> pd.DataFrame:
    """
    For each term (possibly multi-word), tokenize with clean_and_tokenize(),
    look up each token’s vector (if in-vocab), then average them.
    """
    rows = []
    for category, terms in terms_map.items():
        for term in terms:
            tokens = clean_and_tokenize(term)
            if not tokens:
                continue
            # gather only in-vocab token vectors
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            if not vecs:
                continue
            # mean-pool
            mean_vec = np.mean(vecs, axis=0)
            row = {
                'term': term,
                'category': category,
            }
            # unpack into dim0, dim1, …
            for i, val in enumerate(mean_vec):
                row[f'dim{i}'] = float(val)
            rows.append(row)
        # also embed the category name
        cat_tokens = clean_and_tokenize(category)
        if not cat_tokens:
            continue
        cat_vecs = [model.wv[t] for t in cat_tokens if t in model.wv]
        if not cat_vecs:
            continue
        cat_mean_vec = np.mean(cat_vecs, axis=0)
        row = {
            'term': category,
            'category': category,
        }
        for i, val in enumerate(cat_mean_vec):
            row[f'dim{i}'] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def reduce_to_2d(df: pd.DataFrame) -> pd.DataFrame:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[dim_cols].values)
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


def main():
    p = argparse.ArgumentParser(
        description="Visualize CBOW & Skip-gram embeddings with phrase mean-pooling"
    )
    p.add_argument('--terms',     required=True, help="Path to your rcpd_terms.json")
    p.add_argument('--model_dir', required=True, help="Folder with word2vec_*.model files")
    args = p.parse_args()

    cbow_path = os.path.join(args.model_dir, "word2vec_cbow.model")
    skip_path = os.path.join(args.model_dir, "word2vec_skipgram.model")
    for pth in (cbow_path, skip_path, args.terms):
        if not os.path.exists(pth):
            p.error(f"Not found: {pth}")

    ts = datetime.datetime.now().strftime("%m_%d_%H_%M")
    viz_dir = os.path.join(args.model_dir, f"embedding_visualization_{ts}")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Writing visualization HTML into: {viz_dir}\n")

    terms_map   = load_terms(args.terms)
    total_terms = sum(len(v) for v in terms_map.values())
    print(f"Loaded {total_terms} terms across {len(terms_map)} categories\n")

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


if __name__ == "__main__":
    main()
