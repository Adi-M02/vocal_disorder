#!/usr/bin/env python3
"""
Cluster Word2Vec terms with PCA (to 46D) + L2 normalization + KMeans sweep.

Inputs:
  - Gensim Word2Vec/KeyedVectors model
  - Vocab JSON/TXT (JSON supports keys: {"seed_terms"|"terms"|"vocabulary"})

Pipeline:
  1) Load model + intersect with provided vocab
  2) Fit PCA (n_components=46) on the *subset* vectors and transform them
  3) L2-normalize rows (good for cosine-like clustering with k-means)
  4) For K in [k_min..k_max]: run KMeans (n_init, max_iter), collect metrics
     - inertia (within-cluster SSE)
     - silhouette_score (euclidean on unit vectors ~ cosine)
     - davies_bouldin_score (optional sanity)
     - calinski_harabasz_score (optional sanity)
  5) Choose K using silhouette (primary) with a tie-breaker via inertia elbow and
     a light "interpretability" proxy (cluster cohesion)
  6) Save cluster assignments and per-cluster top terms near centroid

Outputs (timestamped directory):
  - tokens_used.txt, oov_terms.txt
  - pca.joblib (fitted PCA(46))
  - Z_46_l2.npz  (tokens + 46D normalized embeddings)
  - k_scan_metrics.csv  (K, inertia, silhouette, db, ch, cohesion_mean, size_imbalance)
  - silhouette_vs_k.png, inertia_vs_k.png (with elbow) and selection_summary.json
  - clusters_K<best>.csv  (token, cluster, sim_to_centroid)
  - cluster_summaries_K<best>.json (per cluster: size, mean_sim, top_terms)

Usage:
  python cluster_pca_kmeans.py \
      --model path/to/model.model \
      --vocab path/to/terms.json \
      --outdir runs/cluster \
      [--k-min 8 --k-max 60] \
      [--n-init 20 --max-iter 500] \
      [--random-state 42] \
      [--top-terms 25]

Notes:
  - PCA is fit on the provided subset by default; if you want a global basis,
    adapt the code to fit on kv.vectors (or add a --fit-on flag like earlier).
  - After L2 normalization, Euclidean distances correspond to cosine distances
    (up to a monotonic transform), making KMeans behave like spherical k-means.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from joblib import dump
import matplotlib.pyplot as plt

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

    JSON accepted keys: seed_terms | terms | vocabulary
    TXT: one term per line
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("seed_terms", "terms", "vocabulary"):
            if key in data and isinstance(data[key], list):
                return [str(t) for t in data[key]]
        raise ValueError(
            f"JSON {path} missing one of keys: seed_terms | terms | vocabulary"
        )
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


def knee_max_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index (into x,y) of the point farthest from the line joining
    the first and last points. Suitable for decreasing inertia curves.
    """
    assert x.ndim == y.ndim == 1 and x.size == y.size >= 3
    x1, y1 = float(x[0]), float(y[0])
    x2, y2 = float(x[-1]), float(y[-1])
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0:
        return int(np.argmax(y))
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    d = num / denom
    return int(np.argmax(d))


def cluster_cohesion(Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """Mean cosine-like similarity of points to their (L2-normalized) centroid.
    Returns (global_mean, per_cluster_mean_dict).
    """
    # Normalize centroids to unit length for cosine-like dot products
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
        # Top terms in this cluster by similarity to centroid
        order = np.argsort(-sims_k)[: top_terms]
        top = [tokens[i] for i in idx[order]]
        out.append({
            "cluster": int(k),
            "size": size,
            "mean_sim": mean_sim,
            "top_terms": top,
        })
    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="PCA(46) + L2 + KMeans sweep for Word2Vec vocab")
    ap.add_argument("--model", required=True, help="Path to Gensim model or KeyedVectors")
    ap.add_argument("--vocab", required=True, help="Path to vocab (JSON or TXT)")
    ap.add_argument("--outdir", required=True, help="Directory for timestamped outputs")

    ap.add_argument("--k-min", type=int, default=8, help="Minimum K to try (default 8)")
    ap.add_argument("--k-max", type=int, default=60, help="Maximum K to try (default 60)")
    ap.add_argument("--n-init", type=int, default=20, help="KMeans n_init (default 20)")
    ap.add_argument("--max-iter", type=int, default=500, help="KMeans max_iter (default 500)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--top-terms", type=int, default=25, help="Top terms per cluster in summary (default 25)")

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

    # Build matrix for subset
    X = kv[in_vocab]  # shape: (n_tokens, emb_dim)

    # PCA -> 46 dims
    logging.info("Fitting PCA(n_components=46) on subset")
    pca = PCA(n_components=46, svd_solver="auto", random_state=args.random_state)
    Z = pca.fit_transform(X)

    # L2-normalize
    Z = normalize(Z, norm="l2", axis=1)

    # Persist PCA transformer and the 46D normalized embedding
    dump(pca, outdir / "pca.joblib")
    np.savez_compressed(outdir / "Z_46_l2.npz", tokens=np.array(in_vocab, dtype=object), Z=Z)

    # K sweep
    Ks = np.arange(int(args.k_min), int(args.k_max) + 1)
    metrics = {
        "K": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
        "cohesion_mean": [],
        "size_imbalance": [],  # std(size)/mean(size)
    }
    label_by_k: Dict[int, np.ndarray] = {}
    kmeans_by_k: Dict[int, KMeans] = {}

    for K in Ks:
        km = KMeans(n_clusters=int(K), n_init=int(args.n_init), max_iter=int(args.max_iter), random_state=args.random_state)
        lab = km.fit_predict(Z)

        inertia = float(km.inertia_)
        # silhouette requires >=2 clusters
        sil = float(silhouette_score(Z, lab, metric="euclidean")) if K > 1 else float("nan")
        try:
            db = float(davies_bouldin_score(Z, lab))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(Z, lab))
        except Exception:
            ch = float("nan")
        # Cohesion + size imbalance
        coh_mean, coh_per = cluster_cohesion(Z, lab, km.cluster_centers_)
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
    import csv
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

    Ks = np.array(metrics["K"], dtype=float)
    inertia = np.array(metrics["inertia"], dtype=float)
    silhouette = np.array(metrics["silhouette"], dtype=float)
    cohesion = np.array(metrics["cohesion_mean"], dtype=float)
    size_imbalance = np.array(metrics["size_imbalance"], dtype=float)

    # Elbow on inertia (decreasing)
    if Ks.size >= 3:
        idx_knee = knee_max_distance(Ks, inertia)
        K_elbow = int(Ks[idx_knee])
    else:
        K_elbow = int(Ks[np.argmin(inertia)])

    # Primary choice: maximize silhouette
    K_sil = int(Ks[np.nanargmax(silhouette)])

    # Interpretability proxy: cohesion high, size imbalance low
    # Build a composite z-score: S = z(sil) + 0.5*z(coh) - 0.25*z(size_imb)
    def _z(a: np.ndarray) -> np.ndarray:
        m, s = np.nanmean(a), np.nanstd(a) + 1e-9
        return (a - m) / s
    comp = _z(silhouette) + 0.5*_z(cohesion) - 0.25*_z(size_imbalance)
    K_comp = int(Ks[np.nanargmax(comp)])

    # Final rule: pick K with silhouette within 95% of best, then closest to elbow; if tie, max composite
    sil_best = np.nanmax(silhouette)
    mask = silhouette >= (0.95 * sil_best)
    if np.any(mask):
        Ks_cand = Ks[mask]
        K_final = int(Ks_cand[np.argmin(np.abs(Ks_cand - K_elbow))])
        # break ties by composite
        if np.sum(mask & (Ks == K_final)) > 1:
            K_final = int(Ks[np.nanargmax(np.where(mask, comp, -np.inf))])
    else:
        # fallback to composite, then silhouette
        K_final = K_comp if not np.isnan(K_comp) else K_sil

    # Plots
    plt.figure(figsize=(7,4))
    plt.plot(Ks, silhouette, marker='o', linewidth=1)
    plt.axvline(K_sil, linestyle='--', label=f"sil max @ {K_sil}")
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K")
    plt.ylabel("Silhouette (euclidean)")
    plt.title("Silhouette vs K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "silhouette_vs_k.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(Ks, inertia, marker='o', linewidth=1)
    plt.axvline(K_elbow, linestyle='--', label=f"elbow @ {K_elbow}")
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K")
    plt.ylabel("Inertia (SSE)")
    plt.title("Inertia vs K (elbow)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "inertia_vs_k.png", dpi=160)
    plt.close()

    # Summaries for the chosen K
    km_best = kmeans_by_k[K_final]
    lab_best = label_by_k[K_final]

    # Save assignments
    import csv
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

    # Decision file
    selection = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "embedding_dim": int(emb_dim),
        "n_terms": int(len(in_vocab)),
        "pca_components": 46,
        "k_range": [int(Ks[0]), int(Ks[-1])],
        "K_elbow_inertia": int(K_elbow),
        "K_silhouette_max": int(K_sil),
        "K_composite": int(K_comp),
        "K_final": int(K_final),
        "metrics_at_final": {
            "silhouette": float(silhouette[Ks == K_final][0]),
            "inertia": float(inertia[Ks == K_final][0]),
            "cohesion_mean": float(cohesion[Ks == K_final][0]),
            "size_imbalance": float(size_imbalance[Ks == K_final][0]),
        },
    }
    (outdir / "selection_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    logging.info(
        f"Chosen K = {selection['K_final']} | sil={selection['metrics_at_final']['silhouette']:.4f} "
        f"| inertia={selection['metrics_at_final']['inertia']:.2f} | cohesion={selection['metrics_at_final']['cohesion_mean']:.4f}"
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
