#!/usr/bin/env python3
"""
UMAP → Ward hierarchical clustering (+ corpus term counts, summaries)

This script reduces embeddings with UMAP, runs Ward hierarchical clustering,
sweeps K (number of clusters) to select a best K using internal indices,
and ALWAYS counts per-cluster term frequencies over a corpus (same behavior
as your KMeans script).

It mirrors the outputs/analysis style you use for KMeans:
  - K scan with silhouette / DBI / CH / cohesion / size imbalance
  - Dendrogram (truncated)
  - clusters_K<best>.csv with sim_to_centroid
  - cluster_summaries_K<best>.json with top terms by sim
  - cluster_term_freqs_K<best>.json with top-N by frequency per cluster
  - cluster_terms_minSim_K<best>.json and cluster_terms_minSim_lines_K<best>.json
  - selection_summary.json (decision record)

UMAP notes:
  - Uses cosine metric in ORIGINAL space by default (good for word2vec).
  - Clustering is performed on z-scored UMAP coordinates using Ward+Euclidean.

fastcluster:
  - If installed, we use fastcluster.linkage_vector(..., method='ward') which is
    faster and more memory efficient than SciPy's linkage with identical results.
  - If not installed, fall back to scipy.cluster.hierarchy.linkage.

Docs source:
  - If --ngram-phraser-dir is provided: use process_ngram_docs(ngram_phraser_dir=...)
  - Else: fall back to process_all_noburp() (no n-gram phrasing)

Example:
  python umap_ward_cluster.py \
    --model /path/to/w2v.model \
    --vocab /path/to/seed_terms.json \
    --outdir runs/umap_ward \
    --umap-dim 12 \
    --k-min 3 --k-max 30 \
    --ngram-phraser-dir /path/to/phrasers \
    --top-freq 50 --min-sim 0.0
"""

from __future__ import annotations
import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import csv
import os
import math
import pickle
import numpy as np

# ---- corpus loaders ----
sys.path.append('../vocal_disorder')
from utils.load_process_ngram_docs import process_ngram_docs
from utils.load_and_process_docs import process_all_noburp

# Headless matplotlib (safe with multiprocessing)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ML stack
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# UMAP
try:
    from umap import UMAP
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires umap-learn to be installed: pip install umap-learn") from e

# Hierarchical clustering: prefer fastcluster, else SciPy
_FASTCLUSTER = False
try:
    import fastcluster  # type: ignore
    _FASTCLUSTER = True
except Exception:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram  # fallback

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
    if ext in {".model", ".w2v"}:
        try:
            m = Word2Vec.load(str(path))
            return m.wv
        except Exception:
            pass
    try:
        kv = KeyedVectors.load(str(path), mmap='r')
        return kv
    except Exception:
        pass
    if ext in {".bin", ".txt", ".vec", ".gz"}:
        try:
            binary = ext == ".bin"
            kv = KeyedVectors.load_word2vec_format(str(path), binary=binary)
            return kv
        except Exception:
            pass
    raise ValueError(f"Unrecognized or unsupported Word2Vec/KeyedVectors file: {path}")


def plot_truncated_dendrogram(Z, last_p: int, outpath: Path, title: str) -> None:
    """Use SciPy's dendrogram for visualization, even if Z was produced by fastcluster."""
    from scipy.cluster.hierarchy import dendrogram as _dendrogram  # ensure the drawer is available
    plt.figure(figsize=(20, 7))
    plt.title(title)
    plt.xlabel('Cluster index (includes counts)')
    plt.ylabel('Distance')
    _dendrogram(Z, truncate_mode='lastp', p=last_p, show_leaf_counts=True,
                leaf_rotation=90., leaf_font_size=8., show_contracted=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def cluster_cohesion(Yz: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """Mean cosine-like similarity of points to their (L2-normalized) centroid in Yz space."""
    cent = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = np.sum(Yz * cent[labels], axis=1)
    means = {}
    for k in np.unique(labels):
        means[int(k)] = float(np.mean(sims[labels == k]))
    return float(np.mean(sims)), means


def summarize_clusters(tokens: List[str], Yz: np.ndarray, labels: np.ndarray,
                       centroids: np.ndarray, top_terms: int = 25) -> List[dict]:
    """Top terms per cluster by similarity to centroid (in Yz)."""
    cent = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims_all = Yz @ cent.T  # (n, K)
    out = []
    for k in range(cent.shape[0]):
        idx = np.where(labels == k)[0]
        size = int(idx.size)
        if size == 0:
            out.append({"cluster": int(k), "size": 0, "mean_sim": 0.0, "top_terms": []})
            continue
        sims_k = sims_all[idx, k]
        mean_sim = float(np.mean(sims_k))
        order = np.argsort(-sims_k)[: top_terms]
        top = [tokens[i] for i in idx[order]]
        out.append({"cluster": int(k), "size": size, "mean_sim": mean_sim, "top_terms": top})
    return out


def compute_centroids(Yz: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """Simple mean centroids in Yz for each cluster label in [1..K] (fcluster convention)."""
    centroids = np.zeros((K, Yz.shape[1]), dtype=np.float64)
    for k in range(1, K + 1):
        idx = np.where(labels == k)[0]
        if idx.size > 0:
            centroids[k - 1] = Yz[idx].mean(axis=0)
    return centroids


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="UMAP → Ward hierarchical clustering (+ corpus term counts)")
    ap.add_argument("--model", required=True, help="Path to Gensim model or KeyedVectors")
    ap.add_argument("--vocab", required=True, help="Path to vocab (JSON or TXT with terms/seed_terms)")
    ap.add_argument("--outdir", required=True, help="Directory for timestamped outputs")

    # UMAP params
    ap.add_argument("--umap-dim", type=int, default=12, help="UMAP target dimension for clustering")
    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--umap-metric", type=str, default="cosine",
                    help="Metric in the ORIGINAL space for UMAP (cosine good for Word2Vec)")
    ap.add_argument("--umap-epochs", type=int, default=None)
    ap.add_argument("--umap-random-state", type=int, default=42)
    ap.add_argument("--densmap", action="store_true")

    # K sweep
    ap.add_argument("--k-min", type=int, default=3, help="Minimum K to try")
    ap.add_argument("--k-max", type=int, default=30, help="Maximum K to try")

    # Reporting
    ap.add_argument("--top-terms", type=int, default=25, help="Top terms per cluster (by sim to centroid)")
    ap.add_argument("--dendro-last-p", type=int, default=100, help="How many last clusters to display in dendrogram")
    ap.add_argument("--min-sim", type=float, default=0.0, help="Minimum sim_to_centroid to include a term (outputs)")

    # Docs source for frequency counting
    ap.add_argument("--ngram-phraser-dir", default=None,
                    help="Dir with trained n-gram phrasers; if omitted, use process_all_noburp() (no n-grams).")
    ap.add_argument("--top-freq", type=int, default=50,
                    help="Top-N terms by frequency per cluster (default 50)")

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

    # Build matrix for subset and normalize for cosine stability in original space
    X = kv[in_vocab].astype(np.float32)  # shape: (n_tokens, emb_dim)
    X = normalize(X, norm="l2", axis=1, copy=True)

    # ----- UMAP reduction -----
    n = X.shape[0]
    umap_dim = int(min(args.umap_dim, max(2, n - 1)))  # ensure feasible
    logging.info(f"Fitting UMAP(n_components={umap_dim}, n_neighbors={args.umap_n_neighbors}, "
                 f"min_dist={args.umap_min_dist}, metric={args.umap_metric}, densmap={args.densmap})")
    reducer = UMAP(
        n_components=umap_dim,
        n_neighbors=min(args.umap_n_neighbors, n-1),
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_epochs=args.umap_epochs,
        random_state=args.umap_random_state,
        densmap=args.densmap,
    )
    Y = reducer.fit_transform(X)  # (n, umap_dim)

    # Z-score before Ward for axis-scale neutrality
    scaler = StandardScaler(with_mean=True, with_std=True)
    Yz = scaler.fit_transform(Y)

    # Persist embedding artifacts
    np.save(outdir / f"Y_umap{umap_dim}.npy", Y)
    with open(outdir / f"umap_reducer_dim{umap_dim}.pkl", "wb") as f:
        pickle.dump(reducer, f)

    # ----- Ward linkage -----
    if _FASTCLUSTER:
        logging.info("Using fastcluster.linkage_vector(..., method='ward') for speed and lower memory.")
        Z = fastcluster.linkage_vector(Yz, method='ward')  # Euclidean implied
    else:
        logging.info("Using SciPy linkage(Yz, method='ward', metric='euclidean'). "
                     "Install 'fastcluster' for faster & more memory-efficient clustering.")
        from scipy.cluster.hierarchy import linkage as _linkage
        Z = _linkage(Yz, method='ward', metric='euclidean')

    # Save linkage for later reuse
    with open(outdir / f"ward_Z_dim{umap_dim}.pkl", "wb") as f:
        pickle.dump(Z, f)

    # Dendrogram (truncated)
    plot_truncated_dendrogram(
        Z, last_p=int(args.dendro_last_p),
        outpath=outdir / f"dendrogram_dim{umap_dim}_last{args.dendro_last_p}.png",
        title=f"Ward dendrogram (UMAP dim={umap_dim}) – last {args.dendro_last_p} merges"
    )

    # ----- K sweep & metrics -----
    from scipy.cluster.hierarchy import fcluster  # ensure available for labels
    Ks = np.arange(int(args.k_min), int(args.k_max) + 1)
    metrics = {"K": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": [],
               "cohesion_mean": [], "size_imbalance": []}
    label_by_k: Dict[int, np.ndarray] = {}
    centroids_by_k: Dict[int, np.ndarray] = {}

    for K in Ks:
        labels = fcluster(Z, int(K), criterion="maxclust")  # labels in 1..K
        # internal metrics on Yz
        try:
            sil = float(silhouette_score(Yz, labels, metric="euclidean")) if K > 1 else float("nan")
        except Exception:
            sil = float("nan")
        try:
            db = float(davies_bouldin_score(Yz, labels))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(Yz, labels))
        except Exception:
            ch = float("nan")

        # cohesion & size balance
        centroids = compute_centroids(Yz, labels, K=K)
        coh_mean, _ = cluster_cohesion(Yz, labels - 1, centroids)  # shift labels to 0..K-1 for indexing
        sizes = np.bincount(labels, minlength=K + 1)[1:].astype(float)  # ignore 0 bin
        size_imb = float(np.std(sizes) / (np.mean(sizes) + 1e-12))

        metrics["K"].append(K)
        metrics["silhouette"].append(sil)
        metrics["davies_bouldin"].append(db)
        metrics["calinski_harabasz"].append(ch)
        metrics["cohesion_mean"].append(coh_mean)
        metrics["size_imbalance"].append(size_imb)

        label_by_k[K] = labels
        centroids_by_k[K] = centroids
        logging.info(f"K={K:>3} | sil={sil:.4f} | DBI={db:.4f} | CH={ch:.1f} | coh={coh_mean:.4f} | size_imb={size_imb:.3f}")

    # Save metrics CSV
    with open(outdir / "k_scan_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K", "silhouette", "davies_bouldin", "calinski_harabasz", "cohesion_mean", "size_imbalance"])
        for i in range(len(Ks)):
            w.writerow([
                int(metrics['K'][i]),
                f"{metrics['silhouette'][i]:.6f}",
                f"{metrics['davies_bouldin'][i]:.6f}",
                f"{metrics['calinski_harabasz'][i]:.6f}",
                f"{metrics['cohesion_mean'][i]:.6f}",
                f"{metrics['size_imbalance'][i]:.6f}",
            ])

    Ks_arr = np.array(metrics["K"], dtype=float)
    sil_arr = np.array(metrics["silhouette"], dtype=float)
    dbi_arr = np.array(metrics["davies_bouldin"], dtype=float)
    ch_arr  = np.array(metrics["calinski_harabasz"], dtype=float)
    coh_arr = np.array(metrics["cohesion_mean"], dtype=float)
    imb_arr = np.array(metrics["size_imbalance"], dtype=float)

    # Selection (no inertia elbow here).
    # Primary: max silhouette. Tie-break: low DBI, high CH, high cohesion, low size_imbalance.
    if Ks_arr.size > 0:
        K_sil = int(Ks_arr[np.nanargmax(sil_arr)])
    else:
        raise SystemExit("No K values provided.")

    def _z(a: np.ndarray) -> np.ndarray:
        m, s = np.nanmean(a), (np.nanstd(a) + 1e-9)
        return (a - m) / s

    comp = _z(sil_arr) - _z(dbi_arr) + 0.5*_z(ch_arr) + 0.5*_z(coh_arr) - 0.25*_z(imb_arr)
    K_comp = int(Ks_arr[np.nanargmax(comp)])
    # Choose final K near the silhouette peak with composite as tie-break
    K_final = K_comp

    # Plots
    plt.figure(figsize=(7,4))
    plt.plot(Ks_arr, sil_arr, marker='o', linewidth=1)
    plt.axvline(K_sil, linestyle='--', label=f"sil max @ {K_sil}")
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K"); plt.ylabel("Silhouette (euclidean)"); plt.title("Silhouette vs K (Ward on UMAP)")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "silhouette_vs_k.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(Ks_arr, dbi_arr, marker='o', linewidth=1)
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K"); plt.ylabel("Davies–Bouldin"); plt.title("DBI vs K (Ward on UMAP)")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "dbi_vs_k.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(Ks_arr, ch_arr, marker='o', linewidth=1)
    plt.axvline(K_final, linestyle=':', label=f"chosen K = {K_final}")
    plt.xlabel("K"); plt.ylabel("Calinski–Harabasz"); plt.title("CH vs K (Ward on UMAP)")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "ch_vs_k.png", dpi=160); plt.close()

    # Summaries for the chosen K
    lab_best = label_by_k[K_final]             # labels in 1..K
    cents = centroids_by_k[K_final]            # (K, dim)
    K = K_final

    # Save assignments (with sim to centroid)
    with open(outdir / f"clusters_K{K}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "cluster", "sim_to_centroid"])
        # normalize centroids for dot-product sim
        cent = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        # shift labels to 0..K-1 for indexing
        lab0 = lab_best - 1
        sims = (Yz @ cent.T)[np.arange(Yz.shape[0]), lab0]
        for t, c, s in zip(in_vocab, lab_best.tolist(), sims.tolist()):
            w.writerow([t, int(c), f"{s:.6f}"])

    # Save cluster summaries by sim
    summaries = summarize_clusters(in_vocab, Yz, lab_best, cents, top_terms=int(args.top_terms))
    (outdir / f"cluster_summaries_K{K}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # -------------------------
    # ALWAYS: Per-cluster term frequency over corpus
    # -------------------------
    if args.ngram_phraser_dir:
        logging.info(f"Loading tokenized+phrased docs with process_ngram_docs(ngram_phraser_dir='{args.ngram_phraser_dir}')")
        docs_iter = process_ngram_docs(ngram_phraser_dir=args.ngram_phraser_dir)
        docs_source = "ngram_phrased"
    else:
        logging.info("No --ngram-phraser-dir provided; falling back to process_all_noburp() (no n-gram phrasing).")
        docs_iter = process_all_noburp(show_progress=True)
        docs_source = "processed_no_ngrams"

    # map term -> (cluster, sim_to_centroid)
    term2cluster = {t: int(c) for t, c in zip(in_vocab, lab_best.tolist())}
    # reuse sims computed above
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

    logging.info(f"Counted cluster-term frequencies over {total_docs:,} docs "
                 f"({total_tokens_matched:,} matched tokens).")

    # Build JSON with top-N by frequency per cluster, filtered by min similarity
    top_n = int(args.top_freq)
    min_sim = float(args.min_sim)
    clusters_out = []
    for k in range(1, K + 1):
        cnt = per_cluster_counts.get(k, Counter())
        # filter by similarity threshold
        filtered_items = [(t, c) for t, c in cnt.items() if term2sim.get(t, -1.0) >= min_sim]
        if not filtered_items:
            logging.warning(f"Cluster {k}: 0 terms meet min-sim={min_sim:.3f}; output will be empty for this cluster.")
        # sort by (-count, -sim, term)
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

    freq_path = outdir / f"cluster_term_freqs_K{K}.json"
    freq_payload = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "K_final": int(K),
        "umap_dim": int(umap_dim),
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

    # -------------------------
    # NEW: Just clusters -> terms with sim_to_centroid >= min_sim
    # -------------------------
    clusters_terms_list = {}
    clusters_terms_lines = {}
    for k in range(1, K + 1):
        idx_k = np.where((lab_best == k) & (sims >= min_sim))[0]
        order = np.argsort(-sims[idx_k])
        idx_k = idx_k[order]
        terms_above = [in_vocab[i] for i in idx_k]
        clusters_terms_list[str(k)] = terms_above
        clusters_terms_lines[str(k)] = "\n".join(terms_above)

    terms_min_sim_list_path = outdir / f"cluster_terms_minSim_K{K}.json"
    terms_min_sim_list_path.write_text(json.dumps(clusters_terms_list, indent=2), encoding="utf-8")
    logging.info(f"Wrote cluster→terms (list) JSON to {terms_min_sim_list_path}")

    terms_min_sim_lines_path = outdir / f"cluster_terms_minSim_lines_K{K}.json"
    terms_min_sim_lines_path.write_text(json.dumps(clusters_terms_lines, indent=2), encoding="utf-8")
    logging.info(f"Wrote cluster→terms (newline) JSON to {terms_min_sim_lines_path}")

    # Decision file
    selection = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "embedding_dim": int(emb_dim),
        "n_terms": int(len(in_vocab)),
        "umap_dim": int(umap_dim),
        "k_range": [int(Ks_arr[0]), int(Ks_arr[-1])],
        "K_silhouette_max": int(K_sil),
        "K_composite": int(K_comp),
        "K_final": int(K),
        "metrics_at_final": {
            "silhouette": float(sil_arr[Ks_arr == K][0]) if K in Ks_arr else float("nan"),
            "davies_bouldin": float(dbi_arr[Ks_arr == K][0]) if K in Ks_arr else float("nan"),
            "calinski_harabasz": float(ch_arr[Ks_arr == K][0]) if K in Ks_arr else float("nan"),
            "cohesion_mean": float(coh_arr[Ks_arr == K][0]) if K in Ks_arr else float("nan"),
            "size_imbalance": float(imb_arr[Ks_arr == K][0]) if K in Ks_arr else float("nan"),
        },
        "docs_source": docs_source,
        "ngram_phraser_dir": args.ngram_phraser_dir,
        "min_sim": min_sim,
        "top_n": top_n,
        "fastcluster_used": bool(_FASTCLUSTER),
    }
    (outdir / "selection_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    logging.info(
        f"Chosen K = {selection['K_final']} | sil={selection['metrics_at_final']['silhouette']:.4f} "
        f"| DBI={selection['metrics_at_final']['davies_bouldin']:.4f} | CH={selection['metrics_at_final']['calinski_harabasz']:.1f} "
        f"| cohesion={selection['metrics_at_final']['cohesion_mean']:.4f} | UMAP D={umap_dim} "
        f"| docs_source={docs_source} | min_sim={min_sim:.3f} | fastcluster={_FASTCLUSTER}"
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
