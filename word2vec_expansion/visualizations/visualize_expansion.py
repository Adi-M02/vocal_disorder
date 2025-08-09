#!/usr/bin/env python3
"""
usage:
  python visualize_expansion.py \
    --model word2vec_expansion/word2vec_08_07_16_41/word2vec_cbow.model \
    --seed_json rcpd_seed_terms.json \
    --topk 15 \
    --min_sim 0.55 \
    --projection pca \
    --output_html expansion_plot.html

Notes:
- Colors seeds by category when seed_json is a dict; otherwise all seeds are one color.
- Edges shown only if similarity >= --min_sim (keeps the figure readable).
- Uses ScatterGL so ~10k points stays responsive.
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.offline import plot as plot_html

# project utils
sys.path.append("../vocal_disorder")
from utils.text_pipeline import process_text
from utils.load_json import load_json


def load_model(path: str) -> Word2Vec:
    return Word2Vec.load(str(Path(path).expanduser()))

def normalize_seed_terms(seeds_obj):
    """Return (seed_terms, seed_category_map, categories_list)."""
    if isinstance(seeds_obj, dict):
        cat_map = {}
        for cat, terms in seeds_obj.items():
            for t in terms:
                key = " ".join(process_text(t))
                cat_map[key] = cat
        seed_terms = list(cat_map.keys())
        categories = sorted(set(cat_map.values()))
        return seed_terms, cat_map, categories
    else:
        seed_terms = [" ".join(process_text(t)) for t in seeds_obj]
        cat_map = {t: "seed" for t in seed_terms}
        categories = ["seed"]
        return seed_terms, cat_map, categories

def most_similar_k(model: Word2Vec, term: str, k: int):
    if term not in model.wv:
        return []
    return model.wv.most_similar(term, topn=k)  # list of (word, cosine)

def build_graph(model: Word2Vec, seed_terms, k: int, min_sim: float):
    """
    Returns:
      nodes: dict term -> vector
      edges: list of (seed, nbr, sim) with sim >= min_sim
    """
    nodes = {}
    edges = []

    # add seeds first
    for s in seed_terms:
        if s in model.wv:
            nodes[s] = model.wv[s]

    # neighbors
    for s in seed_terms:
        nbrs = most_similar_k(model, s, k)
        for n, sim in nbrs:
            if sim < min_sim:
                continue
            # record edge
            edges.append((s, n, float(sim)))
            # record node vectors if present
            if n in model.wv and n not in nodes:
                nodes[n] = model.wv[n]

    return nodes, edges

def project_2d(vectors: np.ndarray, method: str = "pca", random_state: int = 42):
    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(vectors)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            print("[warn] umap-learn not installed; falling back to PCA.", flush=True)
            return PCA(n_components=2, random_state=random_state).fit_transform(vectors)
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                            metric="cosine", random_state=random_state)
        return reducer.fit_transform(vectors)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, learning_rate="auto",
                    init="pca", metric="cosine", random_state=random_state).fit_transform(vectors)
    else:
        raise ValueError(f"Unknown projection method: {method}")

def make_figure(model, nodes, edges, seed_terms, seed_cat_map, categories, projection, output_html):
    # build arrays
    terms = list(nodes.keys())
    X = np.vstack([nodes[t] for t in terms])

    # 2D embedding
    coords = project_2d(X, method=projection)
    xmap = {t: coords[i, 0] for i, t in enumerate(terms)}
    ymap = {t: coords[i, 1] for i, t in enumerate(terms)}

    # degree / type
    seed_set = set(seed_terms)
    deg = {t: 0 for t in terms}
    for s, n, _ in edges:
        if s in deg: deg[s] += 1
        if n in deg: deg[n] += 1

    # Build edge trace (single trace, separated by None for speed)
    ex, ey = [], []
    for s, n, sim in edges:
        if s in xmap and n in xmap:  # safe-guard
            ex += [xmap[s], xmap[n], None]
            ey += [ymap[s], ymap[n], None]
    edge_trace = go.Scattergl(
        x=ex, y=ey,
        mode="lines",
        line=dict(width=1, color="rgba(100,100,100,0.25)"),
        hoverinfo="skip",
        name=f"Edges (n={len(edges)})",
        showlegend=True
    )

    # Node traces – one per seed category (for toggling) + one for neighbors
    traces = [edge_trace]

    # Build neighbor set (not seeds)
    neighbor_only = [t for t in terms if t not in seed_set]
    n_hover = [f"<b>{t}</b><br>type: neighbor<br>deg: {deg[t]}" for t in neighbor_only]
    traces.append(
        go.Scattergl(
            x=[xmap[t] for t in neighbor_only],
            y=[ymap[t] for t in neighbor_only],
            mode="markers",
            marker=dict(size=[max(4, min(12, 4 + np.log1p(deg[t])*2)) for t in neighbor_only],
                        opacity=0.7),
            name=f"Neighbors (n={len(neighbor_only)})",
            text=n_hover,
            hoverinfo="text",
        )
    )

    # Seed categories
    for cat in categories:
        seeds_in_cat = [t for t in seed_terms if seed_cat_map.get(t, "seed") == cat and t in xmap]
        s_hover = []
        for t in seeds_in_cat:
            s_hover.append(f"<b>{t}</b><br>type: seed<br>category: {cat}<br>deg: {deg[t]}")
        traces.append(
            go.Scattergl(
                x=[xmap[t] for t in seeds_in_cat],
                y=[ymap[t] for t in seeds_in_cat],
                mode="markers+text",
                marker=dict(size=[max(6, min(16, 6 + np.log1p(deg[t])*2)) for t in seeds_in_cat]),
                textposition="top center",
                textfont=dict(size=10),
                text=[t if len(seed_terms) <= 150 else "" for t in seeds_in_cat],  # auto-hide labels if too many
                name=f"Seeds: {cat} (n={len(seeds_in_cat)})",
                hoverinfo="text",
                hovertext=s_hover,
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f"Seed Expansion Map — {len(seed_terms)} seeds, {len(terms)} nodes, {len(edges)} edges",
        showlegend=True,
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, t=60, b=10),
        dragmode="pan"
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Save HTML
    plot_html(fig, filename=output_html, auto_open=False, include_plotlyjs="cdn")
    print(f"Wrote interactive plot → {output_html}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to Word2Vec .model")
    ap.add_argument("--seed_json", required=True, help="JSON (list or {cat:[terms]})")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min_sim", type=float, default=0.55, help="Only draw edges ≥ this cosine")
    ap.add_argument("--projection", choices=["pca", "umap", "tsne"], default="pca")
    ap.add_argument("--output_html", default="expansion_plot.html")
    args = ap.parse_args()

    model = load_model(args.model)
    seeds_obj = load_json(args.seed_json)
    seed_terms, seed_cat_map, categories = normalize_seed_terms(seeds_obj)

    nodes, edges = build_graph(model, seed_terms, args.topk, args.min_sim)

    if not nodes:
        print("No nodes found (are seeds OOV?). Exiting.")
        return

    make_figure(model, nodes, edges, seed_terms, seed_cat_map, categories,
                projection=args.projection, output_html=args.output_html)

if __name__ == "__main__":
    main()
