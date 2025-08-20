#!/usr/bin/env python3
"""
UMAP dimension sweep + UMAP grid for HDBSCAN, with DR + clustering metrics.

Part A: Dimension sweep (existing)
  - Trustworthiness T(k), Continuity C(k), Q_NX(k), Jaccard(k)
  - LCMC AUC (1..Kmax)
  - Distance preservation: Spearman, Pearson, Stress-1
  - (optional) quick HDBSCAN stats per dim

Part B: UMAP hyperparameter grid (new)
  - Sweep n_neighbors × min_dist (and optional densmap) at a fixed dimension
  - For each combo (and optional seeds):
      * Same DR metrics (at grid_dim)
      * HDBSCAN metrics: DBCV, noise_frac, n_clusters
      * ARI stability across seeds (if grid_seeds >= 2)
      * Composite score: S = 0.4*mean(T) + 0.2*mean(C) + 0.2*LCMC_AUC + 0.2*(1-Stress)
  - Outputs:
      * grid_metrics.csv (leaderboard sorted by DBCV desc)
      * heatmaps for DBCV and noise across (n_neighbors × min_dist)

Outputs (both parts):
  - metrics.csv (dim sweep)
  - *.png plots for sweep + heatmaps for grid
  - oov_terms.txt, kept_terms.txt
"""

from __future__ import annotations
import argparse, json, math, time, itertools, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.manifold import trustworthiness as skl_trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ARI

# UMAP
try:
    from umap import UMAP
except Exception as e:
    raise SystemExit("Please install umap-learn: pip install umap-learn") from e

# HDBSCAN + DBCV
try:
    import hdbscan
    from hdbscan.validity import validity_index as DBCV
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False


# ---------------- I/O helpers ----------------

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


def load_w2v_vectors(model_path: str, terms: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load Gensim Word2Vec/KeyedVectors and return (X, kept_terms, oov_terms)."""
    from gensim.models import Word2Vec, KeyedVectors
    p = Path(model_path)
    try:
        model = Word2Vec.load(str(p))
        kv = model.wv
    except Exception:
        try:
            kv = KeyedVectors.load(str(p), mmap='r')
        except Exception:
            try:
                kv = KeyedVectors.load_word2vec_format(str(p), binary=p.suffix in {".bin", ".gz"})
            except Exception as e:
                raise SystemExit(f"Could not load model from {model_path}: {e}")

    kept, oov = [], []
    for t in terms:
        (kept if t in kv else oov).append(t)

    if not kept:
        raise SystemExit("No overlap between provided vocab and the model's vocabulary.")
    X = kv[kept]  # shape (n, d)
    return X.astype(np.float32), kept, oov


# ---------------- KNN / neighborhoods ----------------

def knn_indices(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Return neighbor indices (excluding self) shape (n, k)."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nbrs.fit(X)
    _, I = nbrs.kneighbors(X, return_distance=True)
    return I[:, 1:]  # drop self


def neighborhood_overlap_stats(nx: np.ndarray, ny: np.ndarray) -> Tuple[float, float]:
    """
    Average recall Q_NX(k) = |Nx∩Ny|/k and Jaccard = |∩|/|∪| across points.
    nx, ny: (n, k) neighbor index arrays.
    """
    n, k = nx.shape
    recalls, jaccs = [], []
    for i in range(n):
        sx = set(nx[i]); sy = set(ny[i])
        inter = len(sx & sy); uni = len(sx | sy)
        recalls.append(inter / k)
        jaccs.append(inter / uni if uni else 1.0)
    return float(np.mean(recalls)), float(np.mean(jaccs))


# ---------------- Continuity (with reuse of Y ranks) ----------------

def precompute_y_ranks_and_knn(Y: np.ndarray, Kmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - ranks: uint16 matrix, ranks[i, j] = order position of j from i (self rank 0)
      - y_knn_full: (n, Kmax) neighbor indices in Y-space (euclidean)
    """
    n = Y.shape[0]
    Dy = pairwise_distances(Y.astype(np.float32, copy=False), Y.astype(np.float32, copy=False),
                            metric="euclidean", n_jobs=-1)
    ranks = np.empty_like(Dy, dtype=np.uint16)
    for i in range(n):
        order = np.argsort(Dy[i], kind="mergesort")
        r = np.empty(n, dtype=np.uint16)
        r[order] = np.arange(n, dtype=np.uint16)
        ranks[i] = r  # self has rank 0
    y_knn_full = knn_indices(Y, k=Kmax, metric="euclidean")
    return ranks, y_knn_full


def continuity_from_ranks(x_knn_by_k: Dict[int, np.ndarray],
                          ranks: np.ndarray,
                          y_knn_full: np.ndarray,
                          k_list: Iterable[int]) -> Dict[int, float]:
    """
    Compute C(k) for all k in k_list using precomputed Y ranks and Y-KNN.
    """
    n = ranks.shape[0]
    out = {}
    y_knn_sets = None  # lazily build per-k sets
    for k in k_list:
        denom = n * k * (2 * n - 3 * k - 1)
        if denom <= 0:
            out[k] = 1.0
            continue
        # Build Y k-NN sets for this k (lazy)
        yk = y_knn_full[:, :k]
        y_knn_sets = [set(yk[i]) for i in range(n)]
        total = 0.0
        xk = x_knn_by_k[k]
        for i in range(n):
            vx = set(xk[i]) - y_knn_sets[i]
            if not vx:
                continue
            penalty = sum((int(ranks[i, j]) - 1) - k for j in vx)
            total += penalty
        out[k] = 1.0 - (2.0 / denom) * total
    return out


# ---------------- LCMC AUC ----------------

def lcmc_auc(nx_by_k: Dict[int, np.ndarray], ny_by_k: Dict[int, np.ndarray], n: int) -> Tuple[float, int]:
    """
    Compute LCMC(k) for k=1..Kmax and return (AUC, k_at_max), where:
      LCMC(k) = (Q_NX(k) - k/(n-1)) / (1 - k/(n-1))
    """
    Kmax = max(nx_by_k.keys())
    k_vals = np.arange(1, Kmax + 1)
    lcmc_vals = []
    k_at_max, best_v = 1, -1.0
    for k in k_vals:
        qnk, _ = neighborhood_overlap_stats(nx_by_k[k], ny_by_k[k])
        baseline = k / (n - 1)
        denom = (1.0 - baseline) + 1e-12
        lcmc_k = (qnk - baseline) / denom
        lcmc_vals.append(lcmc_k)
        if lcmc_k > best_v:
            best_v, k_at_max = lcmc_k, k
    auc = float(np.trapz(lcmc_vals, k_vals) / Kmax)  # normalized by Kmax
    return auc, k_at_max


# ---------------- Pair sampling + distance stats ----------------

def sample_pairs(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return (m,2) unique pairs with 0 <= i < j < n.
    Tops up until m unique pairs are obtained (capped by n*(n-1)//2).
    """
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)
    max_pairs = n * (n - 1) // 2
    target = min(m, max_pairs)
    pairs = np.empty((0, 2), dtype=np.int32)
    while pairs.shape[0] < target:
        need = target - pairs.shape[0]
        i = rng.integers(0, n - 1, size=need, endpoint=False)
        j = rng.integers(i + 1, n, size=need)  # j < n
        batch = np.stack([i, j], axis=1)
        pairs = np.vstack([pairs, batch])
        pairs = np.unique(pairs, axis=0)
    if pairs.shape[0] > target:
        pairs = pairs[:target]
    return pairs.astype(np.int32, copy=False)


def distance_stats_on_pairs(X: np.ndarray, Y: np.ndarray, pairs: np.ndarray) -> Tuple[float, float, float]:
    """Return (spearman_rho, pearson_r, normalized_stress1)."""
    if pairs.size == 0:
        return 0.0, 0.0, 0.0
    xi, xj = X[pairs[:, 0]], X[pairs[:, 1]]
    yi, yj = Y[pairs[:, 0]], Y[pairs[:, 1]]

    # Cosine distance on X (ensure unit vectors)
    xi_n = xi / (np.linalg.norm(xi, axis=1, keepdims=True) + 1e-9)
    xj_n = xj / (np.linalg.norm(xj, axis=1, keepdims=True) + 1e-9)
    dX = 1.0 - np.sum(xi_n * xj_n, axis=1)

    # Euclidean distance on Y
    dY = np.linalg.norm(yi - yj, axis=1)

    # Spearman via rank corr
    rx = pd.Series(dX).rank(method="average").to_numpy()
    ry = pd.Series(dY).rank(method="average").to_numpy()
    rxm = rx - rx.mean(); rym = ry - ry.mean()
    spearman = float((rxm @ rym) / (np.linalg.norm(rxm) * np.linalg.norm(rym) + 1e-12))

    # Pearson
    xm = dX - dX.mean(); ym = dY - dY.mean()
    pearson = float((xm @ ym) / (np.linalg.norm(xm) * np.linalg.norm(ym) + 1e-12))

    # Normalized stress-1 with optimal linear scaling a
    a = float((dY @ dX) / (dX @ dX + 1e-12))
    num = np.sum((dY - a * dX) ** 2)
    den = np.sum(dY ** 2) + 1e-12
    stress = float(np.sqrt(num / den))
    return spearman, pearson, stress


# ---------------- Plot helpers ----------------

def plot_lines(x, ys_dict, title, ylabel, outpath):
    plt.figure(figsize=(8.2, 5.2))
    for label, y in ys_dict.items():
        plt.plot(x, y, marker='o', label=label)
    plt.title(title)
    plt.xlabel("UMAP target dimension")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if len(ys_dict) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def heatmap(values_df: pd.DataFrame, title: str, outpath: Path,
            vmin: Optional[float] = None, vmax: Optional[float] = None):
    """
    values_df: index=n_neighbors, columns=min_dist (floats), cell=value.
    """
    plt.figure(figsize=(7.2, 5.8))
    data = values_df.to_numpy()
    im = plt.imshow(data, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("min_dist")
    plt.ylabel("n_neighbors")
    plt.xticks(ticks=np.arange(len(values_df.columns)),
               labels=[f"{c:.2f}".rstrip('0').rstrip('.') for c in values_df.columns])
    plt.yticks(ticks=np.arange(len(values_df.index)),
               labels=[str(idx) for idx in values_df.index])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ---------------- HDBSCAN helpers ----------------

def run_hdbscan(Y: np.ndarray,
                min_cluster_size: int,
                min_samples: Optional[int],
                metric: str = "euclidean",
                zscore_before: bool = False) -> Tuple[np.ndarray, int, float, float]:
    """
    Returns:
      labels, n_clusters, noise_frac, dbcv
    """
    if zscore_before:
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-9)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=metric)
    labels = clusterer.fit_predict(Y)
    n = Y.shape[0]
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    try:
        dbcv = float(DBCV(Y, labels, metric=metric))
    except Exception:
        # DBCV undefined if all points noise or single cluster; set to NaN
        dbcv = float("nan")
    noise_frac = n_noise / n
    return labels, n_clusters, noise_frac, dbcv


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="UMAP dim sweep + grid (Word2Vec vocab slice) with DR + HDBSCAN metrics.")
    ap.add_argument("--model", required=True, help="Path to Gensim Word2Vec/KeyedVectors model.")
    ap.add_argument("--vocab_json", required=True, help="Path to JSON with {'seed_terms': [...]}.")

    ap.add_argument("--outdir", required=True, help="Output directory (plots/CSV in timestamped subdir).")

    # --- Dimension sweep ---
    ap.add_argument("--dims", nargs="+", type=int,
                    default=[5,10,15,20,25,30,40,50,60,70,80,90,100],
                    help="UMAP target dimensions to evaluate (sweep).")
    ap.add_argument("--k_list", nargs="+", type=int, default=[15,30,50],
                    help="Neighborhood sizes k for trustworthiness/continuity/overlap.")
    ap.add_argument("--pairs_sample", type=int, default=250000,
                    help="Number of random pairs for distance correlations/stress.")
    ap.add_argument("--random_state", type=int, default=42)

    # UMAP params used for the sweep
    ap.add_argument("--umap_n_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--umap_metric", type=str, default="cosine",
                    help="Metric in the ORIGINAL space for UMAP (cosine recommended for Word2Vec).")
    ap.add_argument("--umap_epochs", type=int, default=None)
    ap.add_argument("--densmap", action="store_true", help="Use densMAP for the sweep.")

    # Quick HDBSCAN during sweep
    ap.add_argument("--hdbscan", action="store_true", help="Quick HDBSCAN sanity per embedding (during sweep).")
    ap.add_argument("--hdb_min_cluster_size", type=int, default=30)
    ap.add_argument("--hdb_min_samples", type=int, default=-1, help="-1 means None")
    ap.add_argument("--hdb_metric", type=str, default="euclidean")
    ap.add_argument("--hdb_zscore_y", action="store_true", help="Z-score Y before HDBSCAN (for comparability).")

    # --- Grid search over UMAP hyperparams (at a fixed dimension) ---
    ap.add_argument("--grid", action="store_true", help="Run UMAP hyperparameter grid for HDBSCAN.")
    ap.add_argument("--grid_dim", type=int, default=25, help="UMAP dimension to use for the grid.")
    ap.add_argument("--grid_n_neighbors", nargs="+", type=int, default=[15, 30, 50],
                    help="List of n_neighbors values to sweep.")
    ap.add_argument("--grid_min_dist", nargs="+", type=float, default=[0.05, 0.10, 0.20],
                    help="List of min_dist values to sweep.")
    ap.add_argument("--grid_densmap", action="store_true", help="Also evaluate densMAP for each grid combo.")
    ap.add_argument("--grid_seeds", type=int, default=1, help="Repeat per combo with different random seeds (>=2 enables ARI).")

    args = ap.parse_args()

    rng = np.random.default_rng(args.random_state)

    # Prepare output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) / f"umap_eval_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load vocab and model
    terms = load_seed_terms(args.vocab_json)
    X_raw, kept, oov = load_w2v_vectors(args.model, terms)
    (out_root / "oov_terms.txt").write_text("\n".join(oov), encoding="utf-8")
    (out_root / "kept_terms.txt").write_text("\n".join(kept), encoding="utf-8")

    # Normalize original vectors for cosine computations
    X = normalize(X_raw, norm="l2", axis=1, copy=True)
    n = X.shape[0]
    print(f"[info] Terms: {n}  (OOV: {len(oov)})  Dim_in: {X.shape[1]}")

    if n < 2:
        raise SystemExit("Need at least 2 terms after OOV filtering.")

    # Sanitize k_list (must be <= n-1)
    k_list = sorted({min(k, n - 1) for k in args.k_list if k >= 1})
    if not k_list:
        k_list = [min(15, n - 1)]
    if max(k_list) != max(args.k_list):
        print(f"[warn] Some k in --k_list exceeded n-1; using clamped list: {k_list}")
    Kmax = max(k_list)

    # Precompute KNN in original space for 1..Kmax (for LCMC)
    print("[info] Building original-space KNN (cosine) for 1..Kmax...")
    x_knn_full = knn_indices(X, k=Kmax, metric="cosine")
    x_knn_by_k_all = {k: x_knn_full[:, :k] for k in range(1, Kmax + 1)}
    x_knn_by_k_eval = {k: x_knn_full[:, :k] for k in k_list}  # subset used often

    # Pre-sample pairs for distance stats
    pairs = sample_pairs(n, args.pairs_sample, rng)

    # ---------- Part A: Dimension sweep ----------
    rows_sweep = []
    if len(args.dims) > 0:
        trust_curves = {f"T@{k}": [] for k in k_list}
        cont_curves  = {f"C@{k}": [] for k in k_list}
        qnx_curves   = {f"Q_NX@{k}": [] for k in k_list}
        jacc_curves  = {f"Jacc@{k}": [] for k in k_list}
        spearman_list, pearson_list, stress_list, lcmc_auc_list = [], [], [], []
        hdbscan_clusters, hdbscan_noise = [], []

        for d in tqdm(args.dims, desc="UMAP dims"):
            umap = UMAP(
                n_components=d,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                n_epochs=args.umap_epochs,
                random_state=args.random_state,
                densmap=args.densmap,
            )
            Y = umap.fit_transform(X)

            # Precompute Y ranks + Y-KNN once per dim
            ranks, y_knn_full = precompute_y_ranks_and_knn(Y, Kmax)
            ny_by_k_all = {k: y_knn_full[:, :k] for k in range(1, Kmax + 1)}

            # DR metrics
            metrics_row = {"dim": d}
            # Trustworthiness (sklearn)
            for k in k_list:
                T = skl_trustworthiness(X, Y, n_neighbors=k, metric="cosine")
                trust_curves[f"T@{k}"].append(T)
                metrics_row[f"T@{k}"] = T
            # Continuity (reusing ranks)
            C_dict = continuity_from_ranks(x_knn_by_k_eval, ranks, y_knn_full, k_list)
            for k in k_list:
                cont_curves[f"C@{k}"].append(C_dict[k])
                metrics_row[f"C@{k}"] = C_dict[k]
            # Overlap metrics
            for k in k_list:
                qnk, jacc = neighborhood_overlap_stats(x_knn_by_k_eval[k], ny_by_k_all[k])
                qnx_curves[f"Q_NX@{k}"].append(qnk)
                jacc_curves[f"Jacc@{k}"].append(jacc)
                metrics_row[f"Q_NX@{k}"] = qnk
                metrics_row[f"Jacc@{k}"] = jacc
            # LCMC AUC up to Kmax
            auc, k_at_max = lcmc_auc(x_knn_by_k_all, ny_by_k_all, n=n)
            lcmc_auc_list.append(auc)
            metrics_row["LCMC_AUC@Kmax"] = auc
            metrics_row["LCMC_k_at_max"] = k_at_max
            # Distance correlations + Stress
            sp, pr, st = distance_stats_on_pairs(X, Y, pairs)
            spearman_list.append(sp); pearson_list.append(pr); stress_list.append(st)
            metrics_row["Spearman_rho"] = sp
            metrics_row["Pearson_r"] = pr
            metrics_row["Stress1_norm"] = st

            # Optional quick HDBSCAN
            if args.hdbscan and _HAS_HDBSCAN:
                min_samples = None if args.hdb_min_samples < 0 else int(args.hdb_min_samples)
                labels, n_clusters, noise_frac, dbcv = run_hdbscan(
                    Y, args.hdb_min_cluster_size, min_samples,
                    metric=args.hdb_metric, zscore_before=args.hdb_zscore_y
                )
                hdbscan_clusters.append(n_clusters)
                hdbscan_noise.append(noise_frac)
                metrics_row["HDBSCAN_clusters"] = n_clusters
                metrics_row["HDBSCAN_noise_frac"] = noise_frac
                metrics_row["HDBSCAN_DBCV"] = dbcv

            rows_sweep.append(metrics_row)

        # Write CSV + plots
        df = pd.DataFrame(rows_sweep).sort_values("dim").reset_index(drop=True)
        csv_path = out_root / "metrics_dim_sweep.csv"
        df.to_csv(csv_path, index=False)
        print(f"[ok] Wrote {csv_path}")

        dims_sorted = sorted(args.dims)
        plot_lines(dims_sorted, trust_curves, "Trustworthiness vs UMAP dim", "Trustworthiness",
                   out_root / "sweep_trustworthiness.png")
        plot_lines(dims_sorted, cont_curves,  "Continuity vs UMAP dim", "Continuity",
                   out_root / "sweep_continuity.png")
        plot_lines(dims_sorted, qnx_curves,   "Neighborhood recall Q_NX(k) vs UMAP dim", "Q_NX(k)",
                   out_root / "sweep_qnx.png")
        plot_lines(dims_sorted, jacc_curves,  "Average Jaccard(k) vs UMAP dim", "Jaccard",
                   out_root / "sweep_jaccard.png")
        plot_lines(dims_sorted, {"LCMC_AUC": lcmc_auc_list}, "LCMC AUC (1..Kmax) vs UMAP dim", "LCMC AUC",
                   out_root / "sweep_lcmc_auc.png")
        plot_lines(dims_sorted, {"Spearman": spearman_list}, "Distance rank correlation vs UMAP dim", "Spearman ρ",
                   out_root / "sweep_spearman.png")
        plot_lines(dims_sorted, {"Pearson":  pearson_list},  "Distance correlation vs UMAP dim", "Pearson r",
                   out_root / "sweep_pearson.png")
        plot_lines(dims_sorted, {"Stress1":  stress_list},   "Normalized stress-1 vs UMAP dim", "Stress-1",
                   out_root / "sweep_stress1.png")
        if args.hdbscan and _HAS_HDBSCAN:
            plot_lines(dims_sorted, {"HDBSCAN clusters": hdbscan_clusters},
                       "HDBSCAN clusters vs UMAP dim", "# clusters",
                       out_root / "sweep_hdbscan_clusters.png")
            plot_lines(dims_sorted, {"HDBSCAN noise frac": hdbscan_noise},
                       "HDBSCAN noise vs UMAP dim", "Noise fraction",
                       out_root / "sweep_hdbscan_noise.png")

    # ---------- Part B: UMAP hyperparameter grid for HDBSCAN ----------
    if args.grid:
        if not _HAS_HDBSCAN:
            raise SystemExit("Grid requested but 'hdbscan' is not available. pip install hdbscan")
        grid_rows = []
        combos = list(itertools.product(sorted(set(args.grid_n_neighbors)),
                                        sorted(set(float(x) for x in args.grid_min_dist)),
                                        [False, True] if args.grid_densmap else [False]))
        print(f"[grid] Evaluating {len(combos)} combos at dim={args.grid_dim} "
              f"with seeds={args.grid_seeds} (densmap={'on' if args.grid_densmap else 'off'})")

        # For composite score we need Kmax DR elements at grid_dim too
        for (nn, md, use_dens) in tqdm(combos, desc="UMAP grid"):
            # Accumulators over seeds
            Ts, Cs, Qs, Jcs = {k: [] for k in k_list}, {k: [] for k in k_list}, {k: [] for k in k_list}, {k: [] for k in k_list}
            lcmc_list, spearman_list, pearson_list, stress_list = [], [], [], []
            dbcv_list, noise_list, nclus_list = [], [], []
            labels_by_seed = []

            for s in range(args.grid_seeds):
                seed = args.random_state + s
                umap = UMAP(
                    n_components=args.grid_dim,
                    n_neighbors=int(nn),
                    min_dist=float(md),
                    metric=args.umap_metric,
                    n_epochs=args.umap_epochs,
                    random_state=seed,
                    densmap=use_dens,
                )
                Y = umap.fit_transform(X)

                # DR metrics @ grid_dim
                ranks, y_knn_full = precompute_y_ranks_and_knn(Y, Kmax)
                ny_by_k_all = {k: y_knn_full[:, :k] for k in range(1, Kmax + 1)}

                for k in k_list:
                    T = skl_trustworthiness(X, Y, n_neighbors=k, metric="cosine")
                    Ts[k].append(T)
                C_dict = continuity_from_ranks({k: x_knn_by_k_all[k] for k in k_list}, ranks, y_knn_full, k_list)
                for k in k_list:
                    Cs[k].append(C_dict[k])
                    qnk, jacc = neighborhood_overlap_stats(x_knn_by_k_all[k], ny_by_k_all[k])
                    Qs[k].append(qnk); Jcs[k].append(jacc)

                auc, _ = lcmc_auc(x_knn_by_k_all, ny_by_k_all, n=n)
                lcmc_list.append(auc)

                sp, pr, st = distance_stats_on_pairs(X, Y, pairs)
                spearman_list.append(sp); pearson_list.append(pr); stress_list.append(st)

                # HDBSCAN metrics
                min_samples = None if args.hdb_min_samples < 0 else int(args.hdb_min_samples)
                labels, n_clusters, noise_frac, dbcv = run_hdbscan(
                    Y, args.hdb_min_cluster_size, min_samples,
                    metric=args.hdb_metric, zscore_before=args.hdb_zscore_y
                )
                labels_by_seed.append(labels)
                nclus_list.append(n_clusters)
                noise_list.append(noise_frac)
                dbcv_list.append(dbcv)

            # Aggregate over seeds
            mean_T = float(np.mean([np.mean(Ts[k]) for k in k_list]))
            mean_C = float(np.mean([np.mean(Cs[k]) for k in k_list]))
            mean_Q = float(np.mean([np.mean(Qs[k]) for k in k_list]))  # FYI
            mean_J = float(np.mean([np.mean(Jcs[k]) for k in k_list])) # FYI
            mean_lcmc = float(np.mean(lcmc_list))
            mean_spearman = float(np.mean(spearman_list))
            mean_pearson = float(np.mean(pearson_list))
            mean_stress = float(np.mean(stress_list))
            mean_dbcv = float(np.nanmean(dbcv_list)) if len(dbcv_list) else float("nan")
            mean_noise = float(np.mean(noise_list)) if len(noise_list) else float("nan")
            mean_nclus = float(np.mean(nclus_list)) if len(nclus_list) else float("nan")

            # ARI stability across seeds (mean pairwise)
            mean_ari = float("nan")
            if args.grid_seeds >= 2:
                aris = []
                for i in range(len(labels_by_seed)):
                    for j in range(i + 1, len(labels_by_seed)):
                        try:
                            aris.append(ARI(labels_by_seed[i], labels_by_seed[j]))
                        except Exception:
                            pass
                if aris:
                    mean_ari = float(np.mean(aris))

            # Composite score (higher is better)
            composite = 0.4 * mean_T + 0.2 * mean_C + 0.2 * mean_lcmc + 0.2 * (1.0 - mean_stress)

            grid_rows.append({
                "n_neighbors": nn,
                "min_dist": md,
                "densmap": use_dens,
                "seeds": args.grid_seeds,
                "mean_T": mean_T,
                "mean_C": mean_C,
                "mean_QNX": mean_Q,
                "mean_Jacc": mean_J,
                "LCMC_AUC": mean_lcmc,
                "Spearman": mean_spearman,
                "Pearson": mean_pearson,
                "Stress1": mean_stress,
                "DBCV": mean_dbcv,
                "noise_frac": mean_noise,
                "n_clusters": mean_nclus,
                "ARI_stability": mean_ari,
                "Composite_S": composite,
            })

        # Leaderboard CSV
        grid_df = pd.DataFrame(grid_rows)
        grid_csv = out_root / "grid_metrics.csv"
        # Sort by DBCV desc, then Composite desc, then noise asc
        grid_df_sorted = grid_df.sort_values(
            by=["DBCV", "Composite_S", "noise_frac"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        grid_df_sorted.to_csv(grid_csv, index=False)
        print(f"[ok] Wrote {grid_csv}")

        # Heatmaps (only for densmap=False subset)
        base_df = grid_df[grid_df["densmap"] == False].copy()
        if not base_df.empty:
            # pivot to (n_neighbors x min_dist)
            def pivot_metric(name: str) -> pd.DataFrame:
                return base_df.pivot_table(index="n_neighbors", columns="min_dist", values=name, aggfunc=np.mean)

            hm_dbcv = pivot_metric("DBCV")
            hm_noise = pivot_metric("noise_frac")
            if not hm_dbcv.empty:
                heatmap(hm_dbcv.sort_index().reindex(sorted(hm_dbcv.columns, key=float), axis=1),
                        f"DBCV heatmap (dim={args.grid_dim}, densmap=False)",
                        out_root / "grid_heatmap_dbcv.png")
            if not hm_noise.empty:
                heatmap(hm_noise.sort_index().reindex(sorted(hm_noise.columns, key=float), axis=1),
                        f"Noise fraction heatmap (dim={args.grid_dim}, densmap=False)",
                        out_root / "grid_heatmap_noise.png")

        # If densmap used, also write a separate leaderboard/heatmaps
        if args.grid_densmap:
            dens_df = grid_df[grid_df["densmap"] == True].copy()
            if not dens_df.empty:
                dens_csv = out_root / "grid_metrics_densmap.csv"
                dens_df.sort_values(by=["DBCV", "Composite_S", "noise_frac"],
                                    ascending=[False, False, True]).to_csv(dens_csv, index=False)
                print(f"[ok] Wrote {dens_csv}")
                def pivot_metric_dm(name: str) -> pd.DataFrame:
                    return dens_df.pivot_table(index="n_neighbors", columns="min_dist", values=name, aggfunc=np.mean)
                hm_dbcv_dm = pivot_metric_dm("DBCV")
                hm_noise_dm = pivot_metric_dm("noise_frac")
                if not hm_dbcv_dm.empty:
                    heatmap(hm_dbcv_dm.sort_index().reindex(sorted(hm_dbcv_dm.columns, key=float), axis=1),
                            f"DBCV heatmap (dim={args.grid_dim}, densmap=True)",
                            out_root / "grid_heatmap_dbcv_densmap.png")
                if not hm_noise_dm.empty:
                    heatmap(hm_noise_dm.sort_index().reindex(sorted(hm_noise_dm.columns, key=float), axis=1),
                            f"Noise fraction heatmap (dim={args.grid_dim}, densmap=True)",
                            out_root / "grid_heatmap_noise_densmap.png")

    print(f"[done] Results in: {out_root}")

if __name__ == "__main__":
    # Quiet down some runtime warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
