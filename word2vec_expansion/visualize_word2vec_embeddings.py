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


def reduce_PCA(df: pd.DataFrame, pca_components: int = 2) -> tuple[pd.DataFrame, dict]:
    if pca_components not in (2, 3):
        raise ValueError(f"pca_components must be 2 or 3, got {pca_components}")

    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    pca = PCA(n_components=pca_components, random_state=42)
    coords = pca.fit_transform(X)

    explained = float(pca.explained_variance_ratio_.sum())
    lost      = 1.0 - explained

    X_recon = pca.inverse_transform(coords)
    mse = np.mean((X - X_recon) ** 2)

    metrics = {
        'explained_variance_ratio': explained,
        'lost_variance':            lost,
        'reconstruction_mse':       mse
    }

    out = df.copy()
    out['x'] = coords[:, 0]
    out['y'] = coords[:, 1]
    if pca_components == 3:
        out['z'] = coords[:, 2]

    return out, metrics


def reduce_tsne(df: pd.DataFrame, n_components: int = 2, **tsne_kwargs) -> tuple[pd.DataFrame, dict]:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    tsne = TSNE(n_components=n_components, random_state=42, **tsne_kwargs)
    coords = tsne.fit_transform(X)

    kl = tsne.kl_divergence_ if hasattr(tsne, 'kl_divergence_') else None
    orig_var = np.var(X, axis=0).sum()
    emb_var  = np.var(coords, axis=0).sum()
    ratio    = emb_var / orig_var

    metrics = {
        'kl_divergence':      kl,
        'variance_ratio':     ratio
    }

    out = df.copy()
    out['x'] = coords[:, 0]
    out['y'] = coords[:, 1]
    if n_components == 3:
        out['z'] = coords[:, 2]

    return out, metrics


def reduce_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    eval_neighbors: int = 10,
    **umap_kwargs
) -> tuple[pd.DataFrame, dict]:
    """
    Runs UMAP, returns (df_with_xyz, metrics_dict) where metrics_dict contains:
      - variance_ratio
      - trustworthiness
      - continuity
      - dist_corr (Spearman)
      - silhouette (requires df['category'])
    """
    # 1) High-dim data
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    # 2) UMAP fit
    um = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        **umap_kwargs
    )
    coords = um.fit_transform(X)

    # 3) variance-ratio proxy
    orig_var = np.var(X, axis=0).sum()
    emb_var  = np.var(coords, axis=0).sum()
    variance_ratio = emb_var / orig_var

    # 4) trustworthiness
    tw = trustworthiness(X, coords, n_neighbors=eval_neighbors)

    # 5) continuity using full low-D ranking
    def continuity_full(X_hd, X_ld, k):
        # neighbors in high-D (including self), only need indices
        nbrs_hd = NearestNeighbors(n_neighbors=k+1).fit(X_hd)
        idx_hd  = nbrs_hd.kneighbors(X_hd, return_distance=False)

        # full low-D distance matrix → full ranking
        D_ld = pairwise_distances(X_ld)
        ranks_ld = np.argsort(D_ld, axis=1)

        N = X_hd.shape[0]
        cont_sum = 0.0

        for i in range(N):
            true_nb = set(idx_hd[i, 1:k+1])        # true neighbors in HD
            missing = true_nb - set(ranks_ld[i, 1:k+1])  # those not in top-k LD

            for j in missing:
                # position of j in the full low-D ranking
                r = np.where(ranks_ld[i] == j)[0][0]
                cont_sum += (r - k)

        return 1 - (2 / (N * k * (2*N - 3*k - 1))) * cont_sum

    cont = continuity_full(X, coords, eval_neighbors)

    # 6) distance-correlation (Spearman over upper triangle)
    D_hd = pairwise_distances(X)
    D_ld = pairwise_distances(coords)
    iu   = np.triu_indices_from(D_hd, k=1)
    dist_corr, _ = spearmanr(D_hd[iu], D_ld[iu])

    # 7) silhouette (requires labels)
    sil = silhouette_score(coords, df['category'].values)

    metrics = {
        'variance_ratio': variance_ratio,
        'trustworthiness': tw,
        'continuity': cont,
        'dist_corr': dist_corr,
        'silhouette': sil
    }

    # 8) attach coords for plotting
    out = df.copy()
    out['x'] = coords[:, 0]
    out['y'] = coords[:, 1]
    if n_components == 3:
        out['z'] = coords[:, 2]

    return out, metrics


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
    ts             = datetime.now().strftime("%m_%d_%H_%M")
    viz_dir        = os.path.join(MODEL_DIR, f"embedding_visualization_{ts}")
    os.makedirs(viz_dir, exist_ok=True)

    # Prepare metrics file
    metrics_path = os.path.join(viz_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        mf.write(f"Embedding reduction metrics — run at {datetime.now()}\n\n")

    # Load terms
    terms_map   = load_terms(TERMS_PATH)
    total_terms = sum(len(v) for v in terms_map.values())
    print(f"Writing visualizations & metrics into: {viz_dir}")
    print(f"Loaded {total_terms} terms across {len(terms_map)} categories\n")

    # Process each model
    for model_path, label in [(os.path.join(MODEL_DIR, "word2vec_cbow.model"), "CBOW"),
                              (os.path.join(MODEL_DIR, "word2vec_skipgram.model"), "SkipGram")]:
        print(f"[{label}] Loading model …")
        model = Word2Vec.load(model_path)
        df    = extract_embeddings(model, terms_map)

        #–– PCA 2D & 3D
        for k in (2, 3):
            print(f"[{label}] Reducing to {k}D via PCA …")
            df_pca, m_pca = reduce_PCA(df, pca_components=k)
            out_html = os.path.join(viz_dir, f"pca{k}d_{label.lower()}.html")
            plot_and_save(df_pca, f"{label} embeddings ({k}D PCA)", out_html)

            with open(metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(f"[{label}] PCA {k}D — explained: {m_pca['explained_variance_ratio']:.2%}, "
                         f"lost: {m_pca['lost_variance']:.2%}, "
                         f"recon MSE: {m_pca['reconstruction_mse']:.6f}\n")

        #–– t-SNE 2D & 3D
        for k in (2, 3):
            print(f"[{label}] Reducing to {k}D via t-SNE …")
            df_tsne, m_tsne = reduce_tsne(df, n_components=k, perplexity=30, learning_rate=200)
            out_html = os.path.join(viz_dir, f"tsne{k}d_{label.lower()}.html")
            plot_and_save(df_tsne, f"{label} embeddings ({k}D t-SNE)", out_html)

            with open(metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(f"[{label}] t-SNE {k}D — KL: {m_tsne['kl_divergence']:.4f}, "
                         f"var ratio: {m_tsne['variance_ratio']:.2%}\n")

        #–– UMAP 2D & 3D
        for k in (2, 3):
            print(f"[{label}] Reducing to {k}D via UMAP …")
            df_um, m_um = reduce_umap(df, n_components=k, n_neighbors=15, min_dist=0.1)
            out_html = os.path.join(viz_dir, f"umap{k}d_{label.lower()}.html")
            plot_and_save(df_um, f"{label} embeddings ({k}D UMAP)", out_html)

            with open(metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(
                    f"[{label}] UMAP {k}D — "
                    f"var ratio:   {m_um['variance_ratio']:.2%}, "
                    f"trust:       {m_um['trustworthiness']:.4f}, "
                    f"cont:        {m_um['continuity']:.4f}, "
                    f"dist_corr:   {m_um['dist_corr']:.4f}, "
                    f"silhouette:  {m_um['silhouette']:.4f}\n"
                )

        print()

    print(f"All done! Metrics written to {metrics_path}")
