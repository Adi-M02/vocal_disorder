#!/usr/bin/env python3
"""
All-in-one pipeline:
  1) PCA scree analysis (per-component variance) with optional "all-but-the-top" (drop-top PCs)
     - Detect elbow on scree (max-distance or curvature)
     - Compute smallest k reaching target cumulative variance on the residual spectrum
     - Save explained_variance.csv, scree_variance.png, decision_pca.json, pca.joblib
     - Save projected embeddings for elbow/target choices
  2) KMeans clustering in the PCA space (L2-normalized)
     - Sweep K, compute metrics (inertia, silhouette, DB, CH, cohesion, size imbalance)
     - Select K via silhouette near elbow + composite score
     - Save metrics/plots, cluster assignments, summaries, per-cluster term frequencies
  3) Interactive 2D Plotly visualization (PCA→2D of the SAME clustering space)

Usage (example):
  python pca_kmeans_all_in_one.py \
      --model path/to/model.model \
      --vocab path/to/terms.json \
      --outdir runs/all_in_one \
      --fit-on subset --fit-topn 200000 \
      --max-components 60 --drop-top 0 --target-variance 0.90 --method max_distance \
      --dim-mode elbow \
      --k-min 8 --k-max 30 --n-init 20 --max-iter 500 --random-state 42 \
      --min-sim 0.2 --top-freq 50

Notes:
  - Vocab JSON accepts keys: {"seed_terms"|"terms"|"vocabulary"}. TXT: one term/line.
  - "All-but-the-top" (drop-top=r) means we *discard* the first r PCs when:
      * searching for the elbow (on the scree of residual PCs),
      * searching for k_target on residual cumulative variance,
      * constructing the reduced embeddings (we slice off the first r PCs).
  - For clustering, we L2-normalize the chosen PCA subspace (works well with cosine-like geometry).
  - The 2D Plotly view projects that *same* clustering space to 2D with PCA.

Requirements:
  pip install gensim scikit-learn joblib matplotlib plotly
"""

from __future__ import annotations
import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import csv
import os

# ---- optional corpus loaders ----
sys.path.append('../vocal_disorder')
try:
    from utils.load_process_ngram_docs import process_ngram_docs  # type: ignore
    from utils.load_and_process_docs import process_all_noburp    # type: ignore
    _HAVE_CORPUS = True
except Exception:
    _HAVE_CORPUS = False

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from joblib import dump

# Headless matplotlib for static plots
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plotly for interactive viz
import plotly.graph_objects as go
from plotly import colors as pcolors
import plotly.io as pio

try:
    from gensim.models import Word2Vec, KeyedVectors
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires gensim to be installed.") from e

# -------------------------
# Utilities
# -------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_outdir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_vocab(path: Path) -> List[str]:
    """Load a vocabulary list from JSON or TXT.

    JSON accepted keys: 'seed_terms', 'terms', or 'vocabulary' containing a list of terms.
    TXT: one term per non-empty, non-#-comment line.
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("seed_terms", "terms", "vocabulary"):
            if key in data and isinstance(data[key], list):
                return [str(t) for t in data[key]]
        raise ValueError(f"JSON {path} missing key: one of seed_terms|terms|vocabulary")
    # Treat as text file
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    if not terms:
        raise ValueError(f"No terms found in text file: {path}")
    return terms


def load_w2v_any(path: Path):
    """Attempt to load a Gensim model or KeyedVectors from common formats."""
    ext = path.suffix.lower()
    # 1) Full Word2Vec model saved by Word2Vec.save
    if ext in {".model", ".w2v"}:
        try:
            m = Word2Vec.load(str(path))
            return m.wv
        except Exception:
            pass
    # 2) KeyedVectors saved by KeyedVectors.save
    try:
        kv = KeyedVectors.load(str(path), mmap='r')
        return kv
    except Exception:
        pass
    # 3) word2vec format (bin or txt)
    if ext in {".bin", ".txt", ".vec"}:
        try:
            binary = ext == ".bin"
            kv = KeyedVectors.load_word2vec_format(str(path), binary=binary)
            return kv
        except Exception:
            pass
    raise ValueError(f"Unrecognized or unsupported Word2Vec/KeyedVectors file: {path}")


def knee_max_distance(y: np.ndarray) -> int:
    """Elbow by maximum distance from the chord joining first/last points (1-based index)."""
    n = len(y)
    if n < 3:
        return n
    x = np.arange(1, n + 1, dtype=float)
    x1, y1 = 1.0, float(y[0])
    x2, y2 = float(n), float(y[-1])
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0:
        return n
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    d = num / denom
    return int(np.argmax(d)) + 1


def _smooth(y: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1 or win > len(y):
        return y.astype(float).copy()
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode='same')


def knee_curvature(y: np.ndarray) -> int:
    """Elbow by maximum discrete second derivative (curvature) on a smoothed scree. (1-based)"""
    n = len(y)
    if n < 3:
        return n
    y_sm = _smooth(y, win=5)
    second = y_sm[:-2] - 2 * y_sm[1:-1] + y_sm[2:]
    return int(np.argmax(second)) + 2


def cluster_cohesion(Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """Mean cosine-like similarity of points to their (L2-normalized) centroid."""
    cent = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = np.sum(Z * cent[labels], axis=1)
    means = {}
    for k in np.unique(labels):
        means[int(k)] = float(np.mean(sims[labels == k]))
    return float(np.mean(sims)), means


def summarize_clusters(tokens: List[str], Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray, top_terms: int = 25) -> List[dict]:
    cent = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims_all = Z @ cent.T  # (n, K)
    out = []
    for k in range(cent.shape[0]):
        idx = np.where(labels == k)[0]
        size = int(idx.size)
        sims_k = sims_all[idx, k]
        mean_sim = float(np.mean(sims_k)) if size > 0 else 0.0
        order = np.argsort(-sims_k)[: top_terms]
        top = [tokens[i] for i in idx[order]]
        out.append({"cluster": int(k), "size": size, "mean_sim": mean_sim, "top_terms": top})
    return out


def _plotly_palette(K: int) -> List[str]:
    """Return ≥K distinct colors by cycling several qualitative palettes."""
    pal = []
    for name in ['Plotly', 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24', 'Set3', 'Pastel', 'Safe']:
        pal.extend(getattr(pcolors.qualitative, name, []))
    if not pal:
        pal = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return [pal[i % len(pal)] for i in range(K)]

# -------------------------
# PCA analysis helper
# -------------------------

def run_pca_analysis(X_fit: np.ndarray, X_subset: np.ndarray, drop_top: int, max_components: int,
                     method: str, target_variance: float, outdir: Path, in_vocab: List[str]) -> Dict[str, object]:
    """Fit PCA on X_fit, compute scree, elbow and target on residual spectrum, and save artifacts.
       Returns dict with keys: pca, table, k_elbow_total, k_target_total, k_eff_elbow, k_eff_target,
       Z_full_subset (projection of subset to full basis), var_ratio, cum, cum_eff
    """
    logging.info(f"Fitting PCA on shape: {X_fit.shape} | max_components={max_components} | drop_top={drop_top}")

    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=42)
    pca.fit(X_fit)

    # Variance tables
    vr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cum = np.cumsum(vr)
    idx = np.arange(1, len(vr) + 1)
    table = np.column_stack([idx, vr, cum])
    np.savetxt(outdir / "explained_variance.csv", table, delimiter=",", fmt=["%d", "%.8f", "%.8f"],
               header="component,var_ratio,cum_var", comments="")

    # Residual spectrum after dropping top PCs
    if drop_top >= max_components:
        logging.warning("--drop-top >= max_components; clamping to max_components-1")
        drop_top = max_components - 1
    var_resid = vr[drop_top:]
    if var_resid.size == 0:
        raise SystemExit("After dropping top PCs, no components remain for elbow detection.")

    if method == "max_distance":
        k_eff_elbow = knee_max_distance(var_resid)
    else:
        k_eff_elbow = knee_curvature(var_resid)
    k_elbow_total = drop_top + k_eff_elbow

    cum_eff = np.cumsum(var_resid)
    k_eff_target = int(np.searchsorted(cum_eff, float(target_variance)) + 1)
    k_eff_target = min(k_eff_target, len(cum_eff))
    k_target_total = drop_top + k_eff_target

    logging.info(f"Elbow k (method={method}, after drop_top={drop_top}): {k_elbow_total} | var_at_elbow={vr[k_elbow_total-1]:.6f}")
    logging.info(f"Target {target_variance:.0%} residual cumulative variance k: {k_target_total} | cum_var_residual={cum_eff[k_eff_target-1]:.6f}")

    # Scree figure
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(1, len(vr) + 1)
    plt.plot(x, vr, marker='o', linewidth=1)
    if drop_top > 0:
        plt.axvspan(1, drop_top, alpha=0.1)
    plt.axvline(k_elbow_total, linestyle='--')
    plt.axhline(vr[k_elbow_total - 1], linestyle='--')
    plt.xlabel("# PCA components")
    plt.ylabel("Per-component explained variance (scree)")
    ttl = "PCA scree plot (per-component variance)"
    if drop_top > 0:
        ttl += f" — drop_top={drop_top}"
    plt.title(ttl)
    plt.tight_layout()
    fig.savefig(outdir / "scree_variance.png", dpi=160)
    plt.close(fig)

    # Persist PCA transformer and projections
    dump(pca, outdir / "pca.joblib")

    Z_full_subset = pca.transform(X_subset)

    # Save reduced embeddings for elbow/target choices
    def _save_z(k_total: int, fname: str):
        k_eff = max(0, int(k_total) - int(drop_top))
        if k_eff <= 0:
            raise SystemExit("Requested k minus drop_top <= 0; nothing to keep.")
        Z = Z_full_subset[:, drop_top: drop_top + k_eff]
        np.savez_compressed(outdir / fname, tokens=np.array(in_vocab, dtype=object), Z=Z)

    _save_z(k_elbow_total, "reduced_embeddings_elbow.npz")
    if k_target_total != k_elbow_total:
        _save_z(k_target_total, "reduced_embeddings_target.npz")

    # Decision JSON (PCA)
    decision = {
        "fit_shape": list(X_fit.shape),
        "max_components": int(max_components),
        "method": method,
        "drop_top": int(drop_top),
        "k_elbow_total": int(k_elbow_total),
        "k_elbow_effective": int(k_eff_elbow),
        "var_at_elbow": float(vr[k_elbow_total - 1]),
        "k_target_total": int(k_target_total),
        "k_target_effective": int(k_eff_target),
        "target_variance_residual": float(target_variance),
        "cum_var_residual_at_target": float(cum_eff[k_eff_target - 1]),
    }
    (outdir / "decision_pca.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    return {
        "pca": pca,
        "table": table,
        "k_elbow_total": k_elbow_total,
        "k_target_total": k_target_total,
        "k_eff_elbow": k_eff_elbow,
        "k_eff_target": k_eff_target,
        "Z_full_subset": Z_full_subset,
        "var_ratio": vr,
        "cum": cum,
        "cum_eff": cum_eff,
        "drop_top": drop_top,
    }

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="PCA scree + KMeans clustering + Plotly 2D viz (all-in-one)")
    ap.add_argument("--model", required=True, help="Path to Gensim model or KeyedVectors")
    ap.add_argument("--vocab", required=True, help="Path to vocab (JSON or TXT)")
    ap.add_argument("--outdir", required=True, help="Directory for timestamped outputs")

    # PCA controls
    ap.add_argument("--fit-on", choices=["subset", "model"], default="subset",
                    help="Fit PCA on 'subset' (your tokens) or entire 'model' vectors")
    ap.add_argument("--fit-topn", type=int, default=200000,
                    help="If --fit-on=model, limit to top-N most frequent tokens (default 200k)")
    ap.add_argument("--max-components", type=int, default=60,
                    help="Maximum PCA components to compute (<= embedding dim)")
    ap.add_argument("--target-variance", type=float, default=0.90,
                    help="Residual cumulative variance target (default 0.90)")
    ap.add_argument("--method", choices=["max_distance", "curvature"], default="max_distance",
                    help="Elbow detection method on the scree curve (default max_distance)")
    ap.add_argument("--drop-top", type=int, default=0,
                    help="Discard the first r principal components (all-but-the-top). 0 disables.")
    ap.add_argument("--dim-mode", choices=["elbow", "target", "fixed"], default="elbow",
                    help="Which PCA dimension to use for clustering: elbow | target | fixed")
    ap.add_argument("--fixed-dim", type=int, default=None,
                    help="If --dim-mode=fixed, use this *total* number of PCs (before drop_top slicing)")

    # KMeans controls
    ap.add_argument("--k-min", type=int, default=3, help="Minimum K to try")
    ap.add_argument("--k-max", type=int, default=30, help="Maximum K to try")
    ap.add_argument("--n-init", type=int, default=20, help="KMeans n_init (default 20)")
    ap.add_argument("--max-iter", type=int, default=500, help="KMeans max_iter (default 500)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--top-terms", type=int, default=25, help="Top terms per cluster in summary (default 25)")

    # Corpus counting & filtering
    ap.add_argument("--ngram-phraser-dir", default=None,
                    help="Directory containing trained n-gram phrasers; if omitted and available, uses process_all_noburp().")
    ap.add_argument("--top-freq", type=int, default=50, help="Top-N terms by frequency per cluster (default 50)")
    ap.add_argument("--min-sim", type=float, default=0.4, help="Min sim_to_centroid to include terms (default 0.0)")

    # Plotly viz
    ap.add_argument("--viz-point-size", type=float, default=6.0, help="Marker size for terms (default 6)")
    ap.add_argument("--viz-opacity", type=float, default=0.85, help="Marker opacity (default 0.85)")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    model_path = Path(args.model)
    vocab_path = Path(args.vocab)
    outdir = ensure_outdir(Path(args.outdir))
    logging.info(f"Output directory: {outdir}")

    # Load model
    logging.info(f"Loading model: {model_path}")
    kv = load_w2v_any(model_path)
    emb_dim = int(kv.vector_size)
    logging.info(f"Model vocab size: {len(kv.key_to_index):,} | dim: {emb_dim}")

    # Load vocab
    terms = load_vocab(vocab_path)
    logging.info(f"Loaded {len(terms):,} terms from {vocab_path}")

    # Intersect with model vocab
    in_vocab = [t for t in terms if t in kv]
    oov = [t for t in terms if t not in kv]
    if not in_vocab:
        raise SystemExit("None of the provided terms are in the model vocabulary.")
    logging.info(f"In-vocab terms: {len(in_vocab):,} | OOV terms: {len(oov):,}")

    # Save OOV & used tokens
    (outdir / "tokens_used.txt").write_text("\n".join(in_vocab), encoding="utf-8")
    if oov:
        (outdir / "oov_terms.txt").write_text("\n".join(oov), encoding="utf-8")

    # Build matrices
    X_subset = kv[in_vocab]  # (n_subset, emb_dim)

    if args.fit_on == "subset":
        X_fit = X_subset
    else:
        logging.info("Fitting PCA on model-wide vectors")
        if args.fit_topn and args.fit_topn < len(kv.key_to_index):
            top_tokens = list(kv.key_to_index.keys())[: int(args.fit_topn)]
            X_fit = kv[top_tokens]
        else:
            X_fit = kv.vectors

    n_fit, emb_dim = X_fit.shape
    max_comps = int(min(max(1, args.max_components), emb_dim))
    drop_top = max(0, int(args.drop_top))

    # ---- PCA analysis ----
    pca_info = run_pca_analysis(
        X_fit=X_fit,
        X_subset=X_subset,
        drop_top=drop_top,
        max_components=max_comps,
        method=str(args.method),
        target_variance=float(args.target_variance),
        outdir=outdir,
        in_vocab=in_vocab,
    )

    # Choose PCA dimension for clustering
    if args.dim_mode == "fixed":
        if args.fixed_dim is None:
            raise SystemExit("--dim-mode=fixed requires --fixed-dim")
        k_total = int(args.fixed_dim)
    elif args.dim_mode == "target":
        k_total = int(pca_info["k_target_total"])  # type: ignore
    else:  # elbow
        k_total = int(pca_info["k_elbow_total"])   # type: ignore

    # Slice the chosen PCA space for clustering
    k_eff = max(0, k_total - drop_top)
    if k_eff <= 0:
        raise SystemExit("Chosen PCA dimension minus drop_top <= 0; nothing to keep for clustering.")

    Z = pca_info["Z_full_subset"][:, drop_top: drop_top + k_eff]  # type: ignore
    # L2-normalize rows
    Z = normalize(Z, norm="l2", axis=1)

    # Persist chosen PCA basis info and Z
    dump(pca_info["pca"], outdir / "pca.joblib")  # already saved, but keep
    np.savez_compressed(outdir / f"Z_selected_l2.npz", tokens=np.array(in_vocab, dtype=object), Z=Z)

    # ---- KMeans sweep ----
    Ks = np.arange(int(args.k_min), int(args.k_max) + 1)
    if Ks.size < 1:
        raise SystemExit("Invalid K range")

    metrics = {"K": [], "inertia": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": [],
               "cohesion_mean": [], "size_imbalance": []}
    label_by_k: Dict[int, np.ndarray] = {}
    kmeans_by_k: Dict[int, KMeans] = {}

    for K in Ks:
        km = KMeans(n_clusters=int(K), n_init=int(args.n_init), max_iter=int(args.max_iter),
                    random_state=args.random_state)
        lab = km.fit_predict(Z)

        inertia = float(km.inertia_)
        sil = float(silhouette_score(Z, lab, metric="euclidean")) if K > 1 else float("nan")
        try:
            db = float(davies_bouldin_score(Z, lab))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(Z, lab))
        except Exception:
            ch = float("nan")
        coh_mean, _ = cluster_cohesion(Z, lab, km.cluster_centers_)
        sizes = np.bincount(lab, minlength=K).astype(float)
        size_imb = float(np.std(sizes) / (np.mean(sizes) + 1e-12))

        metrics["K"].append(K)
        metrics["inertia"].append(inertia)
        metrics["silhouette"].append(sil)
        metrics["davies_bouldin"].append(db)
        metrics["calinski_harabasz"].append(ch)
        metrics["cohesion_mean"].append(coh_mean)
        metrics["size_imbalance"].append(size_imb)

        label_by_k[K] = lab
        kmeans_by_k[K] = km
        logging.info(f"K={K:>3} | inertia={inertia:.2f} | sil={sil:.4f} | coh={coh_mean:.4f} | size_imb={size_imb:.3f}")

    # Save metrics CSV
    with open(outdir / "k_scan_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K","inertia","silhouette","davies_bouldin","calinski_harabasz","cohesion_mean","size_imbalance"])
        for i in range(len(Ks)):
            w.writerow([
                int(metrics['K'][i]),
                f"{metrics['inertia'][i]:.6f}",
                f"{metrics['silhouette'][i]:.6f}",
                f"{metrics['davies_bouldin'][i]:.6f}",
                f"{metrics['calinski_harabasz'][i]:.6f}",
                f"{metrics['cohesion_mean'][i]:.6f}",
                f"{metrics['size_imbalance'][i]:.6f}",
            ])

    Ks_arr = np.array(metrics["K"], dtype=float)
    inertia_arr = np.array(metrics["inertia"], dtype=float)
    silhouette_arr = np.array(metrics["silhouette"], dtype=float)
    cohesion_arr = np.array(metrics["cohesion_mean"], dtype=float)
    size_imb_arr = np.array(metrics["size_imbalance"], dtype=float)

    # Elbow + selection
    if Ks_arr.size >= 3:
        # Elbow on inertia curve (classic)
        idx_knee = np.argmax(np.abs(
            ((inertia_arr[-1] - inertia_arr[0]) * Ks_arr - (Ks_arr[-1] - Ks_arr[0]) * inertia_arr + Ks_arr[-1] * inertia_arr[0] - inertia_arr[-1] * Ks_arr[0]) /
            np.hypot(inertia_arr[-1] - inertia_arr[0], Ks_arr[-1] - Ks_arr[0])
        ))
        K_elbow = int(Ks_arr[idx_knee])
    else:
        K_elbow = int(Ks_arr[np.argmin(inertia_arr)])

    K_sil = int(Ks_arr[np.nanargmax(silhouette_arr)])
    def _z(a: np.ndarray) -> np.ndarray:
        m, s = np.nanmean(a), np.nanstd(a) + 1e-9
        return (a - m) / s
    comp = _z(silhouette_arr) + 0.5*_z(cohesion_arr) - 0.25*_z(size_imb_arr)
    K_comp = int(Ks_arr[np.nanargmax(comp)])

    sil_best = np.nanmax(silhouette_arr)
    mask = silhouette_arr >= (0.95 * sil_best)
    if np.any(mask):
        Ks_cand = Ks_arr[mask]
        K_final = int(Ks_cand[np.argmin(np.abs(Ks_cand - K_elbow))])
        # If tie, prefer higher composite
        tie_idx = np.where((Ks_arr == K_final) & mask)[0]
        if tie_idx.size > 1:
            K_final = int(Ks_arr[np.nanargmax(np.where(mask, comp, -np.inf))])
    else:
        K_final = K_comp if not np.isnan(K_comp) else K_sil

    # Selection plots
    plt.figure(figsize=(7,4))
    plt.plot(Ks_arr, silhouette_arr, marker='o', linewidth=1)
    plt.axvline(K_sil, linestyle='--', label=f"sil max @ {K_sil}")
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K"); plt.ylabel("Silhouette (euclidean)"); plt.title("Silhouette vs K")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "silhouette_vs_k.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(Ks_arr, inertia_arr, marker='o', linewidth=1)
    plt.axvline(K_elbow, linestyle='--', label=f"elbow @ {K_elbow}")
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K"); plt.ylabel("Inertia (SSE)"); plt.title("Inertia vs K (elbow)")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "inertia_vs_k.png", dpi=160); plt.close()

    # Summaries for the chosen K
    km_best = kmeans_by_k[K_final]
    lab_best = label_by_k[K_final]

    # Save assignments (with sim to centroid) and compute sims for reuse
    with open(outdir / f"clusters_K{K_final}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "cluster", "sim_to_centroid"])
        cent = km_best.cluster_centers_
        cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)
        sims = (Z @ cent.T)[np.arange(Z.shape[0]), lab_best]
        for t, c, s in zip(in_vocab, lab_best.tolist(), sims.tolist()):
            w.writerow([t, int(c), f"{s:.6f}"])

    # Save cluster summaries
    summaries = summarize_clusters(in_vocab, Z, lab_best, km_best.cluster_centers_, top_terms=int(args.top_terms))
    (outdir / f"cluster_summaries_K{K_final}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # -------------------------
    # Per-cluster term frequency over corpus (if corpus loaders available)
    # -------------------------
    docs_source = None
    if _HAVE_CORPUS:
        if args.ngram_phraser_dir:
            logging.info(f"Loading tokenized+phrased docs with process_ngram_docs(ngram_phraser_dir='{args.ngram_phraser_dir}')")
            docs_iter = process_ngram_docs(ngram_phraser_dir=args.ngram_phraser_dir)
            docs_source = "ngram_phrased"
        else:
            logging.info("No --ngram-phraser-dir provided; falling back to process_all_noburp() (no n-gram phrasing).")
            docs_iter = process_all_noburp(show_progress=True)
            docs_source = "processed_no_ngrams"

        term2cluster = {t: int(c) for t, c in zip(in_vocab, lab_best.tolist())}
        term2sim = {t: float(s) for t, s in zip(in_vocab, sims.tolist())}

        per_cluster_counts: Dict[int, Counter] = defaultdict(Counter)
        total_docs = 0
        total_tokens_matched = 0
        for doc in docs_iter:
            total_docs += 1
            if not doc:
                continue
            for tok in doc:
                c = term2cluster.get(tok)
                if c is not None:
                    per_cluster_counts[c][tok] += 1
                    total_tokens_matched += 1

        logging.info(f"Counted cluster-term frequencies over {total_docs:,} docs ({total_tokens_matched:,} matched tokens).")

        # Build JSON with top-N by frequency per cluster, filtered by min similarity
        top_n = int(args.top_freq)
        min_sim = float(args.min_sim)
        clusters_out = []
        for k in range(km_best.n_clusters):
            cnt = per_cluster_counts.get(k, Counter())
            filtered_items = [(t, c) for t, c in cnt.items() if term2sim.get(t, -1.0) >= min_sim]
            filtered_items.sort(key=lambda kv: (-kv[1], -term2sim.get(kv[0], 0.0), kv[0]))

            top_struct = [
                {"term": t, "count": int(c), "sim_to_centroid": round(term2sim.get(t, 0.0), 6)}
                for t, c in filtered_items[:top_n]
            ]
            top_lines = [
                f"{t}\t{c}\t{term2sim.get(t, 0.0):.6f}"
                for t, c in filtered_items[:top_n]
            ]

            clusters_out.append({
                "cluster": int(k),
                "size": int(np.sum(lab_best == k)),
                "top_terms_by_frequency": top_struct,
                "top_terms_by_frequency_lines": "\n".join(top_lines),
            })

        freq_path = outdir / f"cluster_term_freqs_K{K_final}.json"
        freq_payload = {
            "model": str(model_path),
            "vocab": str(vocab_path),
            "K_final": int(K_final),
            "pca_components_total": int(k_total),
            "pca_components_effective": int(k_eff),
            "top_n": top_n,
            "min_sim": min_sim,
            "total_docs": int(total_docs),
            "docs_source": docs_source,
            "ngram_phraser_dir": args.ngram_phraser_dir,
            "note": "Counts are token frequencies across all docs; includes only clustered in-vocab terms with sim_to_centroid >= min_sim.",
            "clusters": clusters_out
        }
        freq_path.write_text(json.dumps(freq_payload, indent=2), encoding="utf-8")
        logging.info(f"Wrote per-cluster term frequencies to {freq_path}")
    else:
        logging.info("Corpus loaders not available; skipping frequency counting outputs.")
        min_sim = float(args.min_sim)

    # Always: write cluster→terms above min_sim
    clusters_terms_list = {}
    clusters_terms_lines = {}
    for k in range(km_best.n_clusters):
        idx_k = np.where((lab_best == k) & (sims >= float(args.min_sim)))[0]
        order = np.argsort(-sims[idx_k])
        idx_k = idx_k[order]
        terms_above = [in_vocab[i] for i in idx_k]
        clusters_terms_list[str(k)] = terms_above
        clusters_terms_lines[str(k)] = "\n".join(terms_above)

    (outdir / f"cluster_terms_minSim_K{K_final}.json").write_text(json.dumps(clusters_terms_list, indent=2), encoding="utf-8")
    (outdir / f"cluster_terms_minSim_lines_K{K_final}.json").write_text(json.dumps(clusters_terms_lines, indent=2), encoding="utf-8")

    # -------------------------
    # Interactive 2D Plotly visualization (PCA→2D on the clustering space Z)
    # -------------------------
    logging.info("Creating interactive 2D Plotly visualization of clusters")

    pca2 = PCA(n_components=2, svd_solver="auto", random_state=args.random_state)
    Z2 = pca2.fit_transform(Z)

    Kf = km_best.n_clusters
    palette = _plotly_palette(Kf)

    # Project raw KMeans centroids to 2D
    cent2 = pca2.transform(km_best.cluster_centers_)

    traces = []
    visibility_min = []
    visibility_all = []

    mask_minSim = sims >= float(args.min_sim)
    for k in range(Kf):
        idx = np.where((lab_best == k) & mask_minSim)[0]
        hover = np.stack([
            np.array(in_vocab, dtype=object)[idx],
            np.full(idx.size, k),
            sims[idx]
        ], axis=1) if idx.size else np.empty((0,3), dtype=object)
        traces.append(go.Scattergl(
            x=Z2[idx,0] if idx.size else [],
            y=Z2[idx,1] if idx.size else [],
            mode='markers',
            name=f"c{k} (n={idx.size}) — MinSim",
            marker=dict(size=args.viz_point_size, opacity=args.viz_opacity, color=palette[k]),
            customdata=hover,
            hovertemplate="<b>%{customdata[0]}</b><br>cluster=%{customdata[1]}<br>sim=%{customdata[2]:.3f}<extra></extra>",
            showlegend=True
        ))
        visibility_min.append(True)
        visibility_all.append(False)

    for k in range(Kf):
        idx = np.where(lab_best == k)[0]
        hover = np.stack([
            np.array(in_vocab, dtype=object)[idx],
            np.full(idx.size, k),
            sims[idx]
        ], axis=1) if idx.size else np.empty((0,3), dtype=object)
        traces.append(go.Scattergl(
            x=Z2[idx,0] if idx.size else [],
            y=Z2[idx,1] if idx.size else [],
            mode='markers',
            name=f"c{k} (all n={idx.size})",
            marker=dict(size=args.viz_point_size, opacity=args.viz_opacity, color=palette[k], symbol='circle-open'),
            customdata=hover,
            hovertemplate="<b>%{customdata[0]}</b><br>cluster=%{customdata[1]}<br>sim=%{customdata[2]:.3f}<extra></extra>",
            showlegend=True
        ))
        visibility_min.append(False)
        visibility_all.append(True)

    traces.append(go.Scatter(
        x=cent2[:,0], y=cent2[:,1],
        mode='markers+text',
        marker=dict(size=14, color='black', symbol='x'),
        text=[str(k) for k in range(Kf)],
        textposition='top center',
        name='centroids'
    ))
    visibility_min.append(True)
    visibility_all.append(True)

    updatemenus = [
        dict(
            type='dropdown',
            x=1.02, xanchor='left', y=1.0, yanchor='top',
            buttons=[
                dict(label=f"MinSim ≥ {float(args.min_sim):.3f}", method='update',
                     args=[{'visible': visibility_min},
                           {'title': f"Clusters in 2D (PCA on Z) — MinSim ≥ {float(args.min_sim):.3f}"}]),
                dict(label='All terms', method='update',
                     args=[{'visible': visibility_all},
                           {'title': 'Clusters in 2D (PCA on Z) — All terms'}])
            ]
        )
    ]

    fig = go.Figure(data=traces, layout=go.Layout(
        title=f"Clusters in 2D (PCA on Z) — MinSim ≥ {float(args.min_sim):.3f}",
        xaxis=dict(title='PC1'), yaxis=dict(title='PC2'),
        hovermode='closest',
        updatemenus=updatemenus,
        legend=dict(font=dict(size=10))
    ))

    html_path = outdir / "clusters_2d_plotly.html"
    pio.write_html(fig, file=str(html_path), include_plotlyjs='cdn', full_html=True, auto_open=False)
    logging.info(f"Wrote interactive Plotly HTML: {html_path}")

    # Decision summary JSON (combined)
    selection = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "embedding_dim": int(emb_dim),
        "n_terms": int(len(in_vocab)),
        "pca": {
            "fit_on": args.fit_on,
            "fit_topn": int(args.fit_topn),
            "max_components": int(max_comps),
            "drop_top": int(drop_top),
            "method": str(args.method),
            "target_variance_residual": float(args.target_variance),
            "k_elbow_total": int(pca_info["k_elbow_total"]),
            "k_target_total": int(pca_info["k_target_total"]),
            "chosen_dim_mode": str(args.dim_mode),
            "chosen_k_total": int(k_total),
            "chosen_k_effective": int(k_eff)
        },
        "kmeans": {
            "k_range": [int(Ks_arr[0]), int(Ks_arr[-1])],
            "K_elbow_inertia": int(K_elbow),
            "K_silhouette_max": int(K_sil),
            "K_composite": int(K_comp),
            "K_final": int(K_final),
            "metrics_at_final": {
                "silhouette": float(silhouette_arr[Ks_arr == K_final][0]),
                "inertia": float(inertia_arr[Ks_arr == K_final][0]),
                "cohesion_mean": float(cohesion_arr[Ks_arr == K_final][0]),
                "size_imbalance": float(size_imb_arr[Ks_arr == K_final][0]),
            },
        },
        "plotly_html": str(html_path)
    }
    (outdir / "selection_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    logging.info(
        f"Chosen PCA dim (mode={args.dim_mode}) = {k_total} (eff={k_eff}); "
        f"K = {selection['kmeans']['K_final']} | sil={selection['kmeans']['metrics_at_final']['silhouette']:.4f} | "
        f"inertia={selection['kmeans']['metrics_at_final']['inertia']:.2f} | cohesion={selection['kmeans']['metrics_at_final']['cohesion_mean']:.4f}"
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
