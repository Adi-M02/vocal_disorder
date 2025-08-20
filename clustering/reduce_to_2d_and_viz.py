#!/usr/bin/env python3
"""
PCA-to-3D visualization for a selected vocabulary subset (seed terms).

- Loads Word2Vec/KeyedVectors (.model/.kv or word2vec text/binary)
- Reads a JSON: {"seed_terms": ["term_a", "term_b", ...]}
- Builds a "fit set" for PCA (seeds, optionally +neighbors, optionally +random sample)
- Transforms a "plot set" (seeds, optionally +neighbors)
- Saves:
    * PNG: pca3d_seeds.png  (matplotlib 3D scatter)
    * HTML (if plotly installed): pca3d_seeds.html  (interactive)
    * CSV: pca3d_coords.csv   (term, x, y, z, is_seed, count, cluster_label)
    * TXT: pca3d_variance.txt (PC1..PC3 explained variance & cumulative)
    * tokens_fit.txt / tokens_plot.txt / seed_terms_missing.txt (if any)

Usage (example):
  python pca_3d_seeds_viz.py \
    --model ./w2v/model.model \
    --seed_json ./my_seeds.json \
    --outdir ./viz3d \
    --expand_neighbors_fit 20 \
    --fit_sample_n 20000 \
    --expand_neighbors_show 20 \
    --color_by seed --label_seeds_only --label_top_n 60 \
    --topn 100000 --min_count 5
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (activates 3D projection)
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from gensim.models import Word2Vec, KeyedVectors

# Optional Plotly (interactive). If missing, HTML export is skipped gracefully.
try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===================== Helpers: math / plotting =====================

def l2_unit(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize (preserves cosine geometry)."""
    return normalize(X, norm='l2', axis=1)

def fit_pca_3d(X_fit: np.ndarray) -> PCA:
    pca = PCA(n_components=3, svd_solver="randomized", random_state=RANDOM_STATE)
    pca.fit(X_fit)
    return pca

def label_selection(tokens: List[str], is_seed_flags: List[bool], counts: List[Optional[int]],
                    top_n: int, seeds_only: bool) -> set:
    """
    Decide which points to annotate with text labels (indexes).
    Prefer higher-count terms; seeds break ties.
    """
    n = len(tokens)
    entries = []
    for i in range(n):
        cnt = counts[i] if counts[i] is not None else -1
        entries.append((cnt, 1 if is_seed_flags[i] else 0, i))
    entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = []
    if seeds_only:
        for cnt, seed_flag, i in entries:
            if is_seed_flags[i]:
                chosen.append(i)
                if len(chosen) >= top_n:
                    break
    else:
        chosen = [i for _, _, i in entries[:top_n]]
    return set(chosen)

# ===================== Gensim loading (version-robust) =====================

def _load_any(model_path: str):
    """Try multiple loaders; return either KeyedVectors or Word2Vec."""
    try:
        obj = KeyedVectors.load(model_path, mmap='r')
        return obj
    except Exception:
        pass
    try:
        obj = Word2Vec.load(model_path)
        return obj
    except Exception:
        pass
    for binary in (True, False):
        try:
            obj = KeyedVectors.load_word2vec_format(model_path, binary=binary)
            return obj
        except Exception:
            continue
    raise RuntimeError(f"Could not load model from: {model_path}")

def _as_keyedvectors(obj):
    if isinstance(obj, Word2Vec):
        return obj.wv
    return obj

def _get_tokens(kv: KeyedVectors) -> List[str]:
    if hasattr(kv, "index_to_key"):
        return list(kv.index_to_key)
    if hasattr(kv, "index2word"):
        return list(kv.index2word)
    if hasattr(kv, "index2entity"):
        return list(kv.index2entity)
    raise AttributeError("No token list attribute found on KeyedVectors.")

def _get_count(kv: KeyedVectors, token: str) -> Optional[int]:
    try:
        return kv.get_vecattr(token, "count")
    except Exception:
        pass
    try:
        return kv.vocab[token].count  # type: ignore[attr-defined]
    except Exception:
        return None

def _get_matrix(kv: KeyedVectors) -> np.ndarray:
    if hasattr(kv, "vectors"):
        return kv.vectors
    if hasattr(kv, "syn0"):
        return kv.syn0
    raise AttributeError("No vectors/syn0 found on KeyedVectors.")

def _get_index(kv: KeyedVectors, token: str) -> Optional[int]:
    try:
        return kv.key_to_index[token]
    except Exception:
        pass
    try:
        return kv.vocab[token].index  # type: ignore[attr-defined]
    except Exception:
        return None

def _rows_for_tokens(kv: KeyedVectors, tokens: List[str]) -> np.ndarray:
    """Gather rows aligned to tokens; fall back to per-token get_vector if needed."""
    mat = _get_matrix(kv)
    indices: List[int] = []
    for t in tokens:
        idx = _get_index(kv, t)
        if idx is None:
            indices = []
            break
        indices.append(idx)
    if indices:
        return mat[np.array(indices, dtype=int)]
    return np.vstack([kv.get_vector(t) for t in tokens]).astype(np.float32, copy=False)

def load_gensim_matrix(
    model_path: str,
    restrict_to_topn: Optional[int] = None,
    min_count: Optional[int] = None,
    l2norm: bool = True,
) -> Tuple[np.ndarray, List[str], KeyedVectors]:
    """
    Return (X_all, tokens_all, kv).
    Optionally restrict by top-N and/or min_count (if counts available).
    """
    obj = _load_any(model_path)
    kv = _as_keyedvectors(obj)
    tokens = _get_tokens(kv)

    if min_count is not None:
        filtered: List[str] = []
        any_counts = False
        for t in tokens:
            c = _get_count(kv, t)
            if c is None:
                filtered = tokens
                break
            any_counts = True
            if c >= min_count:
                filtered.append(t)
        tokens = filtered if any_counts else tokens

    if restrict_to_topn is not None:
        tokens = tokens[:restrict_to_topn]

    X = _rows_for_tokens(kv, tokens).astype(np.float32, copy=False)
    if l2norm:
        X = l2_unit(X)
    return X, tokens, kv

# ===================== Seeds, neighbors, sampling =====================

def load_seed_terms(path: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = data.get("seed_terms", [])
    out, seen = [], set()
    for t in seeds:
        if isinstance(t, str):
            tt = t.strip()
            if tt and tt not in seen:
                out.append(tt)
                seen.add(tt)
    return out

def neighbors_for_terms(kv: KeyedVectors, terms: List[str], k: int) -> List[str]:
    """Collect up to k nearest neighbors per term (deduped)."""
    if k <= 0:
        return []
    nbrs = set()
    for t in terms:
        if t in kv:
            try:
                for n, _ in kv.most_similar(t, topn=k):
                    nbrs.add(n)
            except Exception:
                continue
    return list(nbrs)

def build_token_sets(
    tokens_all: List[str],
    kv: KeyedVectors,
    seed_terms: List[str],
    expand_neighbors_fit: int,
    fit_sample_n: int,
    expand_neighbors_show: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (tokens_fit, tokens_plot, missing):
      - tokens_fit: for PCA fitting (seeds [+neighbors] [+sample])
      - tokens_plot: to render (seeds [+neighbors])
      - missing: seeds not in vocab
    """
    token_set = set(tokens_all)
    seeds_in = [t for t in seed_terms if t in token_set]
    missing = [t for t in seed_terms if t not in token_set]

    # Fit set
    fit_set = set(seeds_in)
    if expand_neighbors_fit > 0 and seeds_in:
        fit_set.update(n for n in neighbors_for_terms(kv, seeds_in, expand_neighbors_fit) if n in token_set)
    if fit_sample_n > 0:
        pool = list(token_set - fit_set)
        if pool:
            g = np.random.default_rng(RANDOM_STATE)
            sample = g.choice(pool, size=min(fit_sample_n, len(pool)), replace=False)
            fit_set.update(sample.tolist())
    tokens_fit = list(fit_set)

    # Plot set: seeds + neighbors (context)
    plot_set = set(seeds_in)
    if expand_neighbors_show > 0 and seeds_in:
        plot_set.update(n for n in neighbors_for_terms(kv, seeds_in, expand_neighbors_show) if n in token_set)
    tokens_plot = list(plot_set)

    return tokens_fit, tokens_plot, missing

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="PCA 3D visualization for selected vocab (seed terms).")
    ap.add_argument("--model", "-m", required=True, help="Path to .model/.kv or word2vec format file")
    ap.add_argument("--seed_json", required=True, help="Path to JSON with {'seed_terms': [...]}")

    ap.add_argument("--outdir", "-o", default="./viz3d", help="Directory to write outputs")
    ap.add_argument("--topn", type=int, default=None, help="Restrict to top-N tokens (optional)")
    ap.add_argument("--min_count", type=int, default=None, help="Keep tokens with count >= min_count (if available)")

    # How to build the PCA fit set and the plot set
    ap.add_argument("--expand_neighbors_fit", type=int, default=0, help="Add K nearest neighbors/seed to FIT set (0 disables)")
    ap.add_argument("--fit_sample_n", type=int, default=0, help="Add N random tokens to FIT set for stability (0 disables)")
    ap.add_argument("--expand_neighbors_show", type=int, default=0, help="Add K nearest neighbors/seed to PLOT set (0 disables)")

    # Visual options
    ap.add_argument("--color_by", choices=["seed", "kmeans", "none"], default="seed",
                    help="Color points by 'seed' status, a 3D k-means, or no coloring")
    ap.add_argument("--kmeans_k", type=int, default=12, help="K for 3D k-means if color_by=kmeans")
    ap.add_argument("--label_seeds_only", action="store_true", help="If set, annotate only seeds")
    ap.add_argument("--label_top_n", type=int, default=60, help="Max labels to draw (avoid clutter)")
    ap.add_argument("--figsize", default="10,8", help="Matplotlib fig size WxH (inches), e.g., 10,8")
    ap.add_argument("--dpi", type=int, default=160, help="PNG DPI")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load vectors
    print(f"[info] Loading model: {args.model}")
    X_all, tokens_all, kv = load_gensim_matrix(
        args.model, restrict_to_topn=args.topn, min_count=args.min_count, l2norm=True
    )
    print(f"[info] Matrix: {X_all.shape[0]} tokens x {X_all.shape[1]} dims")

    # Seeds + sets
    seed_terms = load_seed_terms(args.seed_json)
    print(f"[info] Seed terms in JSON: {len(seed_terms)}")
    tokens_fit, tokens_plot, missing = build_token_sets(
        tokens_all, kv, seed_terms,
        expand_neighbors_fit=args.expand_neighbors_fit,
        fit_sample_n=args.fit_sample_n,
        expand_neighbors_show=args.expand_neighbors_show,
    )
    if not tokens_plot:
        raise SystemExit("No tokens to plot (none of the seed terms found).")
    if missing:
        (outdir / "seed_terms_missing.txt").write_text("\n".join(missing), encoding="utf-8")
        print(f"[warn] {len(missing)} seeds not in vocab -> seed_terms_missing.txt")

    # Index mapping and gather matrices
    index = {t: i for i, t in enumerate(tokens_all)}
    X_fit = X_all[[index[t] for t in tokens_fit]]
    X_plot = X_all[[index[t] for t in tokens_plot]]

    # Fit PCA (on FIT set) and transform PLOT set
    pca = fit_pca_3d(X_fit)
    Z = pca.transform(X_plot)  # [n_plot, 3]
    evr = pca.explained_variance_ratio_
    var_text = (f"PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, PC3={evr[2]:.4f}, "
                f"cum={evr[:3].sum():.4f}")
    (outdir / "pca3d_variance.txt").write_text(
        f"Explained variance ratio:\n  {var_text}\n", encoding="utf-8"
    )

    # Metadata per point
    seed_set = set(seed_terms)
    is_seed = [t in seed_set for t in tokens_plot]
    counts = [_get_count(kv, t) for t in tokens_plot]

    # Optional KMeans color (visual only; 3D cluster labels are for visual grouping)
    cluster_labels = None
    if args.color_by == "kmeans":
        if args.kmeans_k <= 1 or args.kmeans_k >= len(tokens_plot):
            print("[warn] Invalid kmeans_k for this plot size; falling back to 'seed' coloring.")
            args.color_by = "seed"
        else:
            km = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=RANDOM_STATE)
            cluster_labels = km.fit_predict(Z)

    # Save CSV of coordinates + metadata
    csv_path = outdir / "pca3d_coords.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "x", "y", "z", "is_seed", "count", "cluster_label"])
        for i, t in enumerate(tokens_plot):
            cl = int(cluster_labels[i]) if cluster_labels is not None else ""
            w.writerow([t, float(Z[i,0]), float(Z[i,1]), float(Z[i,2]),
                        int(is_seed[i]), counts[i] if counts[i] is not None else "", cl])
    print(f"[info] Wrote CSV: {csv_path}")

    # -------- Matplotlib 3D PNG --------
    w,h = (float(x) for x in args.figsize.split(","))
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_subplot(111, projection='3d')

    # Choose colors/labels
    if args.color_by == "kmeans" and cluster_labels is not None:
        sc = ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=cluster_labels, s=12, alpha=0.85, cmap="tab20")
        # overlay seeds with black edge
        if any(is_seed):
            idx_seed = [i for i,b in enumerate(is_seed) if b]
            ax.scatter(Z[idx_seed,0], Z[idx_seed,1], Z[idx_seed,2],
                       c=[cluster_labels[i] for i in idx_seed], s=40, alpha=0.95,
                       cmap="tab20", edgecolors="black", linewidths=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01, label="KMeans cluster (3D)")
    elif args.color_by == "seed":
        idx_ctx = [i for i,b in enumerate(is_seed) if not b]
        if idx_ctx:
            ax.scatter(Z[idx_ctx,0], Z[idx_ctx,1], Z[idx_ctx,2],
                       c="#9bbcd1", s=10, alpha=0.45, label="context")
        idx_seed = [i for i,b in enumerate(is_seed) if b]
        if idx_seed:
            ax.scatter(Z[idx_seed,0], Z[idx_seed,1], Z[idx_seed,2],
                       c="#e4572e", s=40, alpha=0.95, label="seed")
        ax.legend(loc="upper left")
    else:
        ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=12, alpha=0.8)

    # Labels (keep few)
    choose = label_selection(tokens_plot, is_seed, counts, top_n=args.label_top_n, seeds_only=args.label_seeds_only)
    for i in choose:
        ax.text(Z[i,0], Z[i,1], Z[i,2], tokens_plot[i], fontsize=8)

    ax.set_title(f"PCA to 3D of Selected Vocab (fit on chosen set)\n{var_text}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    png_path = outdir / "pca3d_seeds.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=args.dpi)
    plt.close()
    print(f"[info] Wrote PNG: {png_path}")

    # -------- Optional Plotly 3D HTML --------
    if PLOTLY_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame({
            "term": tokens_plot,
            "x": Z[:,0],
            "y": Z[:,1],
            "z": Z[:,2],
            "is_seed": ["seed" if b else "context" for b in is_seed],
            "count": [c if c is not None else np.nan for c in counts],
            "cluster": cluster_labels if cluster_labels is not None else None
        })
        color_col = None
        if args.color_by == "kmeans" and cluster_labels is not None:
            color_col = "cluster"
        elif args.color_by == "seed":
            color_col = "is_seed"

        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color=color_col,
            hover_name="term",
            hover_data={"count":True, "is_seed":True, "x":False, "y":False, "z":False},
            title=f"PCA 3D: Selected Vocabulary — {var_text}",
            opacity=0.9
        )
        sizes = np.where(np.array(is_seed), 6, 3)  # bigger markers for seeds
        fig.update_traces(marker={'size': sizes})
        html_path = outdir / "pca3d_seeds.html"
        pio.write_html(fig, str(html_path), include_plotlyjs="cdn", auto_open=False)
        print(f"[info] Wrote HTML: {html_path}")
    else:
        print("[info] Plotly not installed; skipping interactive HTML. (pip install plotly)")

    # Persist token lists
    (outdir / "tokens_fit.txt").write_text("\n".join(tokens_fit), encoding="utf-8")
    (outdir / "tokens_plot.txt").write_text("\n".join(tokens_plot), encoding="utf-8")
    print(f"[done] Outputs saved in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
