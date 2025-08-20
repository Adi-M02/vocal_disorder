#!/usr/bin/env python3
"""
PCA-50 -> (t-SNE | UMAP) -> 2D visualization for selected vocab (seed terms).

- Loads Word2Vec/KeyedVectors (.model/.kv or word2vec text/binary)
- Reads JSON: {"seed_terms": ["term_a", "term_b", ...]}
- Fits PCA (50D) on a configurable FIT set (seeds ± neighbors ± random sample)
- Applies manifold (t-SNE or UMAP) to 2D on the PLOT set (seeds ± neighbors)
- Saves outputs to a TIMESTAMPED DIR inside --outdir:
    * PNG: manifold2d.png
    * HTML (if plotly installed): manifold2d.html
    * CSV: manifold2d_coords.csv (term,x,y,is_seed,count,cluster_label)
    * TXT: pca50_variance.txt
    * tokens_fit.txt / tokens_plot.txt / seed_terms_missing.txt (if any)

Example:
  python pca50_to_manifold2d.py \
    --model ./w2v/model.model \
    --seed_json ./my_seeds.json \
    --outdir ./viz \
    --method umap --umap_neighbors 30 --umap_min_dist 0.05 --umap_metric cosine \
    --expand_neighbors_fit 20 --fit_sample_n 20000 --expand_neighbors_show 20 \
    --color_by seed --label_seeds_only --label_top_n 60 \
    --topn 100000 --min_count 5
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# Optional imports
try:
    from sklearn.manifold import TSNE
    SK_TSNE_AVAILABLE = True
except Exception:
    SK_TSNE_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from gensim.models import Word2Vec, KeyedVectors

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------ Utils ------------------------------

def l2_unit(X: np.ndarray) -> np.ndarray:
    return normalize(X, norm='l2', axis=1)

def timestamped_subdir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def label_selection(tokens: List[str], is_seed_flags: List[bool], counts: List[Optional[int]],
                    top_n: int, seeds_only: bool) -> set:
    n = len(tokens)
    entries = []
    for i in range(n):
        cnt = counts[i] if counts[i] is not None else -1
        entries.append((cnt, 1 if is_seed_flags[i] else 0, i))
    entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = []
    if seeds_only:
        for _, seed_flag, i in entries:
            if is_seed_flags[i]:
                chosen.append(i)
                if len(chosen) >= top_n: break
    else:
        chosen = [i for _, _, i in entries[:top_n]]
    return set(chosen)

# -------------------- Gensim loading (robust) ----------------------

def _load_any(model_path: str):
    try:
        obj = KeyedVectors.load(model_path, mmap='r'); return obj
    except Exception: pass
    try:
        obj = Word2Vec.load(model_path); return obj
    except Exception: pass
    for binary in (True, False):
        try:
            obj = KeyedVectors.load_word2vec_format(model_path, binary=binary); return obj
        except Exception:
            continue
    raise RuntimeError(f"Could not load model from: {model_path}")

def _as_kv(obj):
    return obj.wv if isinstance(obj, Word2Vec) else obj

def _get_tokens(kv: KeyedVectors) -> List[str]:
    if hasattr(kv, "index_to_key"): return list(kv.index_to_key)
    if hasattr(kv, "index2word"):   return list(kv.index2word)
    if hasattr(kv, "index2entity"): return list(kv.index2entity)
    raise AttributeError("No token list attribute found on KeyedVectors.")

def _get_count(kv: KeyedVectors, token: str) -> Optional[int]:
    try: return kv.get_vecattr(token, "count")
    except Exception: pass
    try: return kv.vocab[token].count  # type: ignore[attr-defined]
    except Exception: return None

def _get_matrix(kv: KeyedVectors) -> np.ndarray:
    if hasattr(kv, "vectors"): return kv.vectors
    if hasattr(kv, "syn0"):    return kv.syn0
    raise AttributeError("No vectors/syn0 found on KeyedVectors.")

def _get_index(kv: KeyedVectors, token: str) -> Optional[int]:
    try: return kv.key_to_index[token]
    except Exception: pass
    try: return kv.vocab[token].index  # type: ignore[attr-defined]
    except Exception: return None

def _rows_for_tokens(kv: KeyedVectors, tokens: List[str]) -> np.ndarray:
    mat = _get_matrix(kv)
    idxs: List[int] = []
    for t in tokens:
        i = _get_index(kv, t)
        if i is None:
            idxs = []
            break
        idxs.append(i)
    if idxs:
        return mat[np.array(idxs, dtype=int)]
    return np.vstack([kv.get_vector(t) for t in tokens]).astype(np.float32, copy=False)

def load_gensim_matrix(
    model_path: str,
    restrict_to_topn: Optional[int] = None,
    min_count: Optional[int] = None,
    l2norm: bool = True,
) -> Tuple[np.ndarray, List[str], KeyedVectors]:
    obj = _load_any(model_path)
    kv = _as_kv(obj)
    tokens = _get_tokens(kv)
    if min_count is not None:
        filt, any_counts = [], False
        for t in tokens:
            c = _get_count(kv, t)
            if c is None:
                filt = tokens; break
            any_counts = True
            if c >= min_count: filt.append(t)
        tokens = filt if any_counts else tokens
    if restrict_to_topn is not None:
        tokens = tokens[:restrict_to_topn]
    X = _rows_for_tokens(kv, tokens).astype(np.float32, copy=False)
    if l2norm:
        X = l2_unit(X)
    return X, tokens, kv

# ------------------------ Seeds & sets -----------------------------

def load_seed_terms(path: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = data.get("seed_terms", [])
    out, seen = [], set()
    for t in seeds:
        if isinstance(t, str):
            tt = t.strip()
            if tt and tt not in seen:
                out.append(tt); seen.add(tt)
    return out

def neighbors_for_terms(kv: KeyedVectors, terms: List[str], k: int) -> List[str]:
    if k <= 0: return []
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
    token_set = set(tokens_all)
    seeds_in = [t for t in seed_terms if t in token_set]
    missing = [t for t in seed_terms if t not in token_set]

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

    plot_set = set(seeds_in)
    if expand_neighbors_show > 0 and seeds_in:
        plot_set.update(n for n in neighbors_for_terms(kv, seeds_in, expand_neighbors_show) if n in token_set)
    tokens_plot = list(plot_set)

    return tokens_fit, tokens_plot, missing

# ------------------------- PCA & Manifold --------------------------

def fit_pca(X_fit: np.ndarray) -> PCA:
    pca = PCA(n_components=15, svd_solver="randomized", random_state=RANDOM_STATE)
    pca.fit(X_fit)
    return pca

def run_tsne(Z50_plot: np.ndarray, perplexity: float, learning_rate: float,
             early_exaggeration: float, n_iter: int) -> np.ndarray:
    if not SK_TSNE_AVAILABLE:
        raise SystemExit("scikit-learn TSNE not available. Install scikit-learn >= 1.2.")
    n = Z50_plot.shape[0]
    # Basic guard: perplexity must be < (n-1)/3
    max_perp = max(5.0, (n - 1) / 3.0 - 1e-6)
    if perplexity >= max_perp:
        print(f"[warn] perplexity={perplexity} too high for n={n}; capping to {max_perp:.1f}")
        perplexity = max_perp
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate=learning_rate,
        early_exaggeration=early_exaggeration, init="pca", n_iter=n_iter,
        random_state=RANDOM_STATE, verbose=0,
    )
    return tsne.fit_transform(Z50_plot)

def run_umap(Z50_plot: np.ndarray, n_neighbors: int, min_dist: float, metric: str) -> np.ndarray:
    if not UMAP_AVAILABLE:
        raise SystemExit("umap-learn not available. pip install umap-learn")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=RANDOM_STATE, verbose=False
    )
    return reducer.fit_transform(Z50_plot)

# ----------------------------- Main -------------------------------

def parse_figsize(s: str) -> Tuple[float, float]:
    w, h = s.split(",")
    return float(w), float(h)

def main():
    ap = argparse.ArgumentParser(description="PCA-50 -> (t-SNE | UMAP) -> 2D viz for selected vocab.")
    ap.add_argument("--model", "-m", required=True, help="Path to .model/.kv or word2vec format file")
    ap.add_argument("--seed_json", required=True, help="Path to JSON with {'seed_terms': [...]}")

    ap.add_argument("--outdir", "-o", default="./viz", help="Base output directory (a timestamped subfolder is created)")
    ap.add_argument("--topn", type=int, default=None, help="Restrict to top-N tokens (optional)")
    ap.add_argument("--min_count", type=int, default=None, help="Keep tokens with count >= min_count (if available)")

    # Build sets
    ap.add_argument("--expand_neighbors_fit", type=int, default=0, help="Add K nearest neighbors/seed to FIT set")
    ap.add_argument("--fit_sample_n", type=int, default=0, help="Add N random tokens to FIT set")
    ap.add_argument("--expand_neighbors_show", type=int, default=0, help="Add K nearest neighbors/seed to PLOT set")

    # Manifold choice & params
    ap.add_argument("--method", choices=["tsne", "umap"], default="umap", help="Manifold method after PCA-50")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (if method=tsne)")
    ap.add_argument("--tsne_lr", type=float, default=200.0, help="t-SNE learning rate (if method=tsne)")
    ap.add_argument("--tsne_ee", type=float, default=12.0, help="t-SNE early exaggeration (if method=tsne)")
    ap.add_argument("--tsne_iter", type=int, default=1000, help="t-SNE iterations (if method=tsne)")

    ap.add_argument("--umap_neighbors", type=int, default=30, help="UMAP n_neighbors (if method=umap)")
    ap.add_argument("--umap_min_dist", type=float, default=0.05, help="UMAP min_dist (if method=umap)")
    ap.add_argument("--umap_metric", type=str, default="cosine", help="UMAP metric (e.g., cosine, euclidean)")

    # Visual options
    ap.add_argument("--color_by", choices=["seed", "kmeans", "none"], default="seed", help="Color by seed, kmeans (2D), or none")
    ap.add_argument("--kmeans_k", type=int, default=12, help="K for 2D k-means if color_by=kmeans")
    ap.add_argument("--label_seeds_only", action="store_true", help="Annotate only seed terms")
    ap.add_argument("--label_top_n", type=int, default=60, help="Max labels to draw")
    ap.add_argument("--figsize", default="10,8", help="Matplotlib fig size WxH, e.g., 10,8")
    ap.add_argument("--dpi", type=int, default=170, help="PNG DPI")
    args = ap.parse_args()

    base_outdir = Path(args.outdir)
    outdir = timestamped_subdir(base_outdir)
    print(f"[info] Writing outputs to: {outdir}")

    # Load vectors
    print(f"[info] Loading model: {args.model}")
    X_all, tokens_all, kv = load_gensim_matrix(
        args.model, restrict_to_topn=args.topn, min_count=args.min_count, l2norm=True
    )
    print(f"[info] Matrix: {X_all.shape[0]} tokens x {X_all.shape[1]} dims")

    # Seeds & token sets
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

    # PCA-50: fit on FIT set, transform PLOT set
    pca = fit_pca(X_fit)
    Z50_fit = pca.transform(X_fit)
    Z50_plot = pca.transform(X_plot)
    evr = pca.explained_variance_ratio_
    cev = np.cumsum(evr)
    (outdir / "pca50_variance.txt").write_text(
        "Explained variance ratio (first 10 of 50):\n  "
        + ", ".join(f"{x:.4f}" for x in evr[:10])
        + f"\nCumulative (50 comps): {cev[-1]:.4f}\n",
        encoding="utf-8"
    )

    # Manifold 2D on the PLOT set (already PCA-50 reduced)
    if args.method == "tsne":
        Y = run_tsne(Z50_plot, args.perplexity, args.tsne_lr, args.tsne_ee, args.tsne_iter)
        method_title = f"PCA-50 → t-SNE-2D (perp={args.perplexity:g})"
    else:
        Y = run_umap(Z50_plot, args.umap_neighbors, args.umap_min_dist, args.umap_metric)
        method_title = f"PCA-50 → UMAP-2D (n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})"

    # Metadata per point
    seed_set = set(seed_terms)
    is_seed = [t in seed_set for t in tokens_plot]
    counts = [_get_count(kv, t) for t in tokens_plot]

    # Optional KMeans (visual only)
    cluster_labels = None
    if args.color_by == "kmeans":
        if args.kmeans_k <= 1 or args.kmeans_k >= len(tokens_plot):
            print("[warn] Invalid kmeans_k; falling back to 'seed' coloring.")
            args.color_by = "seed"
        else:
            km = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=RANDOM_STATE)
            cluster_labels = km.fit_predict(Y)

    # Save CSV
    csv_path = outdir / "manifold2d_coords.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "x", "y", "is_seed", "count", "cluster_label"])
        for i, t in enumerate(tokens_plot):
            cl = int(cluster_labels[i]) if cluster_labels is not None else ""
            w.writerow([t, float(Y[i,0]), float(Y[i,1]), int(is_seed[i]),
                        counts[i] if counts[i] is not None else "", cl])
    print(f"[info] Wrote CSV: {csv_path}")

    # ---- Matplotlib PNG ----
    w,h = parse_figsize(args.figsize)
    plt.figure(figsize=(w,h))
    if args.color_by == "kmeans" and cluster_labels is not None:
        sc = plt.scatter(Y[:,0], Y[:,1], c=cluster_labels, s=18, alpha=0.85, cmap="tab20")
        if any(is_seed):
            idx_seed = [i for i,b in enumerate(is_seed) if b]
            plt.scatter(Y[idx_seed,0], Y[idx_seed,1], c=[cluster_labels[i] for i in idx_seed],
                        s=50, alpha=0.95, cmap="tab20", edgecolors="black", linewidths=0.6)
        plt.colorbar(sc, label="KMeans cluster (2D)")
    elif args.color_by == "seed":
        idx_ctx = [i for i,b in enumerate(is_seed) if not b]
        if idx_ctx:
            plt.scatter(Y[idx_ctx,0], Y[idx_ctx,1], c="#9bbcd1", s=14, alpha=0.45, label="context")
        idx_seed = [i for i,b in enumerate(is_seed) if b]
        if idx_seed:
            plt.scatter(Y[idx_seed,0], Y[idx_seed,1], c="#e4572e", s=50, alpha=0.95, label="seed")
        plt.legend(loc="best", frameon=True)
    else:
        plt.scatter(Y[:,0], Y[:,1], s=16, alpha=0.8)

    # Labels
    chosen = label_selection(tokens_plot, is_seed, counts, top_n=args.label_top_n, seeds_only=args.label_seeds_only)
    for i in chosen:
        plt.text(Y[i,0], Y[i,1], tokens_plot[i], fontsize=8, ha="left", va="bottom")

    plt.title(method_title + f"\nPCA-50 cumulative variance: {cev[-1]:.3f}")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.25)
    png_path = outdir / "manifold2d.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=args.dpi)
    plt.close()
    print(f"[info] Wrote PNG: {png_path}")

    # ---- Optional Plotly HTML ----
    if PLOTLY_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame({
            "term": tokens_plot,
            "x": Y[:,0], "y": Y[:,1],
            "is_seed": ["seed" if b else "context" for b in is_seed],
            "count": [c if c is not None else np.nan for c in counts],
            "cluster": cluster_labels if cluster_labels is not None else None
        })
        color_col = None
        if args.color_by == "kmeans" and cluster_labels is not None: color_col = "cluster"
        elif args.color_by == "seed": color_col = "is_seed"
        fig = px.scatter(
            df, x="x", y="y", color=color_col,
            hover_name="term",
            hover_data={"count":True, "is_seed":True, "x":False, "y":False},
            title=method_title + f" — PCA-50 cum var={cev[-1]:.3f}",
            opacity=0.9
        )
        sizes = np.where(np.array(is_seed), 10, 5)
        fig.update_traces(marker={'size': sizes})
        html_path = outdir / "manifold2d.html"
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
