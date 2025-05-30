import sys
import os
import json
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import umap
import plotly.express as px

# transformers for BERT
from transformers import AutoTokenizer, AutoModel

# utility tokenizer
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize

# ————————————————————————————————————————————————
# 1) term loading (unchanged)
def load_terms(path: str) -> dict[str, list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {cat.replace('_', ' '): [t.replace('_', ' ') for t in terms]
            for cat, terms in terms_map.items()}

# ————————————————————————————————————————————————
# 2) generic extractor for any embed_fn

def extract_embeddings(terms_map: dict[str, list[str]], embed_fn) -> pd.DataFrame:
    rows = []
    for category, terms in terms_map.items():
        # include the category itself as a “term” at the end
        for term in terms + [category]:
            # 1) clean & tokenize with your custom rules
            tokens = clean_and_tokenize(term)
            if not tokens:
                continue

            # 2) join and embed non unigram terms
            text_to_embed = " ".join(tokens)
            vec = embed_fn(text_to_embed)

            if vec is None:
                print(f"Warning: embedding for term '{term}' returned None, skipping.")
                continue

            # 4) record
            row = {"term": term, "category": category}
            for i, val in enumerate(vec):
                row[f"dim{i}"] = float(val)
            rows.append(row)

    return pd.DataFrame(rows)

# ————————————————————————————————————————————————
# 3) dimensionality reduction routines

def reduce_PCA(df: pd.DataFrame, pca_components: int = 2) -> tuple[pd.DataFrame, dict]:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    pca = PCA(n_components=pca_components, random_state=42)
    coords = pca.fit_transform(X)

    explained = float(pca.explained_variance_ratio_.sum())
    lost      = 1.0 - explained
    X_recon   = pca.inverse_transform(coords)
    mse       = np.mean((X - X_recon) ** 2)

    metrics = {
        'explained_variance_ratio': explained,
        'lost_variance':            lost,
        'reconstruction_mse':       mse
    }

    out = df.copy()
    out['x'], out['y'] = coords[:,0], coords[:,1]
    if pca_components == 3:
        out['z'] = coords[:,2]
    return out, metrics


def reduce_tsne(df: pd.DataFrame, n_components: int = 2, **tsne_kwargs) -> tuple[pd.DataFrame, dict]:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X = df[dim_cols].values

    tsne   = TSNE(n_components=n_components, random_state=42, **tsne_kwargs)
    coords = tsne.fit_transform(X)

    kl    = tsne.kl_divergence_ if hasattr(tsne, 'kl_divergence_') else None
    orig_var = np.var(X, axis=0).sum()
    emb_var  = np.var(coords, axis=0).sum()
    ratio    = emb_var / orig_var

    metrics = {
        'kl_divergence':  kl,
        'variance_ratio': ratio
    }

    out = df.copy()
    out['x'], out['y'] = coords[:,0], coords[:,1]
    if n_components == 3:
        out['z'] = coords[:,2]
    return out, metrics


def reduce_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    eval_neighbors: int = 10,
    **umap_kwargs
) -> tuple[pd.DataFrame, dict]:
    dim_cols = [c for c in df.columns if c.startswith('dim')]
    X        = df[dim_cols].values

    um       = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        **umap_kwargs
    )
    coords = um.fit_transform(X)

    orig_var = np.var(X, axis=0).sum()
    emb_var  = np.var(coords, axis=0).sum()
    var_ratio = emb_var / orig_var
    tw       = trustworthiness(X, coords, n_neighbors=eval_neighbors)

    # continuity full ranking
    def continuity_full(X_hd, X_ld, k):
        nbrs_hd = NearestNeighbors(n_neighbors=k+1).fit(X_hd)
        idx_hd  = nbrs_hd.kneighbors(X_hd, return_distance=False)
        D_ld    = pairwise_distances(X_ld)
        ranks_ld = np.argsort(D_ld, axis=1)
        N = X_hd.shape[0]
        cont_sum = 0.0
        for i in range(N):
            true_nb = set(idx_hd[i,1:k+1])
            missing = true_nb - set(ranks_ld[i,1:k+1])
            for j in missing:
                r = np.where(ranks_ld[i] == j)[0][0]
                cont_sum += (r - k)
        return 1 - (2/(N*k*(2*N-3*k-1))) * cont_sum

    cont      = continuity_full(X, coords, eval_neighbors)
    D_hd      = pairwise_distances(X)
    D_ld      = pairwise_distances(coords)
    iu        = np.triu_indices_from(D_hd, k=1)
    dist_corr, _ = spearmanr(D_hd[iu], D_ld[iu])
    sil       = silhouette_score(coords, df['category'])

    metrics = {
        'variance_ratio': var_ratio,
        'trustworthiness': tw,
        'continuity': cont,
        'dist_corr': dist_corr,
        'silhouette': sil
    }

    out = df.copy()
    out['x'], out['y'] = coords[:,0], coords[:,1]
    if n_components == 3:
        out['z'] = coords[:,2]
    return out, metrics

# ————————————————————————————————————————————————
# 4) plotting utility
def plot_and_save(df: pd.DataFrame, title: str, out_html: str):
    if 'z' in df.columns:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='category', hover_data=['term'], title=title, width=800, height=600)
    else:
        fig = px.scatter(df, x='x', y='y', color='category', hover_data=['term'], title=title, width=800, height=600)
    fig.write_html(out_html)
    print(f" → saved {out_html}")

# ————————————————————————————————————————————————
# 5) BERT embedder factory
def bert_embedder(model_name_or_path: str):
    tok   = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    def fn(term: str):
        toks = tok(term, return_tensors='pt', truncation=True)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        h = out.hidden_states[-1].squeeze(0).mean(dim=0).numpy()
        return h
    return fn

# ————————————————————————————————————————————————
if __name__ == "__main__":
    SELF_DIR   = os.path.dirname(os.path.abspath(__file__))
    TERMS_PATH = 'rcpd_terms.json'
    terms_map  = load_terms(TERMS_PATH)

    MODELS = [
        ("finetuned base BERT", lambda: bert_embedder('/local/disk2/not_backed_up/amukundan/research/vocal_disorder/vocab_expansion_with_other_models/finetuned_base_bert/bert-base')),
        # ("BERT-base",      lambda: bert_embedder('bert-base-uncased')),
        # ("BERTweet",       lambda: bert_embedder('vinai/bertweet-base')),
        # ("ClinicalBERT",   lambda: bert_embedder('emilyalsentzer/Bio_ClinicalBERT')),
        # ("PubMedBERT",     lambda: bert_embedder('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')),
    ]

    ts      = datetime.now().strftime("%m_%d_%H_%M")
    OUTDIR  = os.path.join(SELF_DIR, f"multi_model_viz_{ts}")
    os.makedirs(OUTDIR, exist_ok=True)

    metrics_path = os.path.join(OUTDIR, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        mf.write(f"Embedding reduction metrics — run at {datetime.now()}\n\n")

    for label, fn_factory in MODELS:
        print(f"[{label}] loading…")
        embed_fn = fn_factory()
        df       = extract_embeddings(terms_map, embed_fn)

        # PCA 2D & 3D
        for k in (2, 3):
            print(f"[{label}] PCA {k}D…")
            df_pca, m_pca = reduce_PCA(df, pca_components=k)
            out_html      = os.path.join(OUTDIR, f"pca{k}d_{label.replace(' ', '_')}.html")
            plot_and_save(df_pca, f"{label} embeddings ({k}D PCA)", out_html)
            with open(metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(
                    f"[{label}] PCA {k}D — explained: {m_pca['explained_variance_ratio']:.2%}, "
                    f"lost: {m_pca['lost_variance']:.2%}, "
                    f"recon MSE: {m_pca['reconstruction_mse']:.6f}\n"
                )

        # t-SNE 2D & 3D
        for k in (2, 3):
            print(f"[{label}] t-SNE {k}D…")
            df_tsne, m_tsne = reduce_tsne(df, n_components=k, perplexity=30, learning_rate=200)
            out_html        = os.path.join(OUTDIR, f"tsne{k}d_{label.replace(' ', '_')}.html")
            plot_and_save(df_tsne, f"{label} embeddings ({k}D t-SNE)", out_html)
            with open(metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(
                    f"[{label}] t-SNE {k}D — KL: {m_tsne['kl_divergence']:.4f}, "
                    f"var ratio: {m_tsne['variance_ratio']:.2%}\n"
                )

        # UMAP 2D & 3D
        for k in (2, 3):
            print(f"[{label}] UMAP {k}D…")
            df_um, m_um = reduce_umap(df, n_components=k, n_neighbors=15, min_dist=0.1)
            out_html     = os.path.join(OUTDIR, f"umap{k}d_{label.replace(' ', '_')}.html")
            plot_and_save(df_um, f"{label} embeddings ({k}D UMAP)", out_html)
            with open(metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(
                    f"[{label}] UMAP {k}D — var ratio:   {m_um['variance_ratio']:.2%}, "
                    f"trust:       {m_um['trustworthiness']:.4f}, "
                    f"cont:        {m_um['continuity']:.4f}, "
                    f"dist_corr:   {m_um['dist_corr']:.4f}, "
                    f"silhouette:  {m_um['silhouette']:.4f}\n"
                )

        print()

    print(f"All done! Metrics written to {metrics_path}")
