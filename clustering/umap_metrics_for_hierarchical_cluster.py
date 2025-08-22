#!/usr/bin/env python3
"""
UMAP dimension sweep (no clustering)
------------------------------------

What it does
- Loads Word2Vec/KeyedVectors
- Filters to your provided vocab (seed_terms JSON)
- Runs UMAP for a list of target dimensions
- Computes DR metrics per dim:
    * Trustworthiness T(k)  (for k in k_list)
    * Continuity C(k)       (for k in k_list)
    * Neighborhood recall Q_NX(k) and Jaccard(k)
    * LCMC AUC (1..Kmax)
    * Distance preservation: Spearman (rank corr), Pearson, normalized Stress-1
- Recommends a best dimension via composite score
- Saves:
    * metrics_dim_sweep.csv
    * Per-metric vs-dimension plots
    * Per-dim embeddings: Y_dim{d}.npy and embeddings_dim{d}.tsv (terms + coords)
    * Best embedding: Y_best.npy and embeddings_best.tsv
    * recommended.txt

Why cosine for UMAP metric?
- For Word2Vec-like embeddings, cosine in ORIGINAL space is a strong default.

Next step (outside this file)
- Load the chosen Y (e.g., Y_best.npy) and run Ward hierarchical clustering
  with scipy.linkage(Y, method="ward", metric="euclidean") and fcluster.
"""

from __future__ import annotations
import argparse, json, time, math
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.manifold import trustworthiness as skl_trustworthiness
from sklearn.metrics import pairwise_distances

from umap import UMAP


# ---------------- I/O helpers ----------------

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
    X = kv[kept].astype(np.float32)
    return X, kept, oov


# ---------------- KNN / neighborhoods ----------------

def knn_indices(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Return neighbor indices (excluding self) shape (n, k)."""
    from sklearn.neighbors import NearestNeighbors
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


# ---------------- Continuity (precompute Y ranks once) ----------------

def precompute_y_ranks_and_knn(Y: np.ndarray, Kmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - ranks: uint16 matrix, ranks[i, j] = order position of j from i (self rank 0)
      - y_knn_full: (n, Kmax) neighbor indices in Y-space (euclidean)
    """
    from sklearn.neighbors import NearestNeighbors
    n = Y.shape[0]
    # Pairwise distances in Y for ranks
    Dy = pairwise_distances(Y.astype(np.float32, copy=False), Y.astype(np.float32, copy=False),
                            metric="euclidean", n_jobs=-1)
    ranks = np.empty_like(Dy, dtype=np.uint16)
    for i in range(n):
        order = np.argsort(Dy[i], kind="mergesort")
        r = np.empty(n, dtype=np.uint16)
        r[order] = np.arange(n, dtype=np.uint16)
        ranks[i] = r  # self has rank 0
    # KNN in Y
    nbrs = NearestNeighbors(n_neighbors=Kmax + 1, metric="euclidean", n_jobs=-1)
    nbrs.fit(Y)
    _, I = nbrs.kneighbors(Y, return_distance=True)
    y_knn_full = I[:, 1:]  # drop self
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
    for k in k_list:
        denom = n * k * (2 * n - 3 * k - 1)
        if denom <= 0:
            out[k] = 1.0
            continue
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
        j = rng.integers(i + 1, n, size=need)
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


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="UMAP dimension sweep with DR metrics (no clustering)")
    ap.add_argument("--model", required=True, help="Path to Gensim Word2Vec/KeyedVectors model.")
    ap.add_argument("--vocab_json", required=True, help="Path to JSON with {'seed_terms': [...]}.")

    ap.add_argument("--outdir", required=True, help="Output directory (plots/CSV in timestamped subdir).")

    ap.add_argument("--dims", nargs="+", type=int,
                    default=[4,6,8,12,16,20,25,30],
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

    args = ap.parse_args()
    rng = np.random.default_rng(args.random_state)

    # Prepare output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) / f"umap_dim_sweep_{ts}"
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

    # Precompute KNN in original space for 1..Kmax (for LCMC and overlaps)
    print("[info] Building original-space KNN (cosine) for 1..Kmax...")
    x_knn_full = knn_indices(X, k=Kmax, metric="cosine")
    x_knn_by_k_all = {k: x_knn_full[:, :k] for k in range(1, Kmax + 1)}
    x_knn_by_k_eval = {k: x_knn_full[:, :k] for k in k_list}  # subset used often

    # Pre-sample pairs for distance stats
    pairs = sample_pairs(n, args.pairs_sample, rng)

    # ---------- Dimension sweep ----------
    rows_sweep = []

    trust_curves = {f"T@{k}": [] for k in k_list}
    cont_curves  = {f"C@{k}": [] for k in k_list}
    qnx_curves   = {f"Q_NX@{k}": [] for k in k_list}
    jacc_curves  = {f"Jacc@{k}": [] for k in k_list}
    spearman_list, pearson_list, stress_list, lcmc_auc_list = [], [], [], []

    best_dim, best_score = None, -1e9
    best_Y = None

    for d in args.dims:
        print(f"[dim] Fitting UMAP to {d} dims (nn={args.umap_n_neighbors}, min_dist={args.umap_min_dist}, metric={args.umap_metric})")
        umap = UMAP(
            n_components=d,
            n_neighbors=min(args.umap_n_neighbors, n-1),
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

        # DR metrics (averages across k_list where relevant)
        metrics_row = {"dim": d}

        # Trustworthiness (sklearn)
        T_vals = {}
        for k in k_list:
            T = skl_trustworthiness(X, Y, n_neighbors=k, metric="cosine")
            trust_curves[f"T@{k}"].append(T)
            metrics_row[f"T@{k}"] = T
            T_vals[k] = T

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

        rows_sweep.append(metrics_row)

        # Save per-dim embeddings for later clustering
        np.save(out_root / f"Y_dim{d}.npy", Y)
        # TSV with terms for convenience
        dfY = pd.DataFrame(Y, columns=[f"Dim{i+1}" for i in range(Y.shape[1])])
        dfY.insert(0, "term", kept)
        dfY.to_csv(out_root / f"embeddings_dim{d}.tsv", sep="\t", index=False)

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
    plot_lines(dims_sorted, {"Spearman": [r["Spearman_rho"] for r in rows_sweep]}, "Distance rank correlation vs UMAP dim", "Spearman ρ",
               out_root / "sweep_spearman.png")
    plot_lines(dims_sorted, {"Pearson":  [r["Pearson_r"] for r in rows_sweep]},  "Distance correlation vs UMAP dim", "Pearson r",
               out_root / "sweep_pearson.png")
    plot_lines(dims_sorted, {"Stress1":  [r["Stress1_norm"] for r in rows_sweep]},   "Normalized stress-1 vs UMAP dim", "Stress-1",
               out_root / "sweep_stress1.png")

    # --------- Recommend a dimension (composite) ---------
    # Score = 0.4*mean(T) + 0.2*mean(C) + 0.2*LCMC_AUC + 0.2*(1 - Stress)
    # (averages across provided k_list)
    comps = []
    for r in rows_sweep:
        T_mean = float(np.mean([r[f"T@{k}"] for k in k_list]))
        C_mean = float(np.mean([r[f"C@{k}"] for k in k_list]))
        S = 0.4*T_mean + 0.2*C_mean + 0.2*r["LCMC_AUC@Kmax"] + 0.2*(1.0 - r["Stress1_norm"])
        comps.append((r["dim"], S))
    # Normalize to break very tight ties using stress (lower better) and LCMC
    comps_sorted = sorted(comps, key=lambda x: x[1], reverse=True)
    rec_dim = comps_sorted[0][0]

    # Load and save the best embedding explicitly as Y_best
    bestY = np.load(out_root / f"Y_dim{rec_dim}.npy")
    np.save(out_root / "Y_best.npy", bestY)
    dfY_best = pd.DataFrame(bestY, columns=[f"Dim{i+1}" for i in range(bestY.shape[1])])
    dfY_best.insert(0, "term", kept)
    dfY_best.to_csv(out_root / "embeddings_best.tsv", sep="\t", index=False)

    # Rationale
    rdict = next(r for r in rows_sweep if r["dim"] == rec_dim)
    msg = [
        f"Recommended UMAP dim={rec_dim}",
        f"- Composite score = {dict(comps)[rec_dim]:.6f}",
        f"- mean T(k)      = {np.mean([rdict[f'T@{k}'] for k in k_list]):.4f}",
        f"- mean C(k)      = {np.mean([rdict[f'C@{k}'] for k in k_list]):.4f}",
        f"- LCMC AUC       = {rdict['LCMC_AUC@Kmax']:.4f}",
        f"- Stress-1       = {rdict['Stress1_norm']:.4f}",
        "",
        "Next: run Ward clustering on Y_best.npy (or embeddings_best.tsv).",
        "Example:",
        "  from scipy.cluster.hierarchy import linkage, fcluster",
        "  import numpy as np",
        "  Y = np.load('Y_best.npy')",
        "  Z = linkage(Y, method='ward', metric='euclidean')",
        "  labels = fcluster(Z, K, criterion='maxclust')",
    ]
    (out_root / "recommended.txt").write_text("\n".join(msg), encoding="utf-8")
    print("\n[recommendation]")
    print("\n".join(msg))

    print(f"\n[done] Results in: {out_root}")

if __name__ == "__main__":
    main()
