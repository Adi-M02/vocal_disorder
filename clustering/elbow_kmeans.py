#!/usr/bin/env python3
"""
Cluster Word2Vec terms with PCA (to N dims) + L2 normalization + KMeans sweep,
then ALWAYS count per-cluster term frequencies over a corpus.

Docs source:
  - If --ngram-phraser-dir is provided: use process_ngram_docs(ngram_phraser_dir=...)
  - Else: fall back to process_all_noburp() (no n-gram phrasing)

Outputs (in the timestamped run dir under --outdir) include:
  - cluster_term_freqs_K<best>.json
      - per cluster:
        - top_terms_by_frequency (structured list of dicts)
        - top_terms_by_frequency_lines (newline string: "term\\tcount\\tsim")
  - cluster_terms_minSim_K<best>.json
      - cluster -> [terms with sim >= --min-sim]
  - cluster_terms_minSim_lines_K<best>.json
      - cluster -> "term1\\nterm2\\n..."  (newline-formatted for readability)
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

# ---- corpus loaders ----
sys.path.append('../vocal_disorder')
from utils.load_process_ngram_docs import process_ngram_docs
from utils.load_and_process_docs import process_all_noburp

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from joblib import dump

# Headless matplotlib (safe with multiprocessing)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
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
    if ext in {".bin", ".txt", ".vec"}:
        try:
            binary = ext == ".bin"
            kv = KeyedVectors.load_word2vec_format(str(path), binary=binary)
            return kv
        except Exception:
            pass
    raise ValueError(f"Unrecognized or unsupported Word2Vec/KeyedVectors file: {path}")


def knee_max_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Index of point farthest from the line joining first/last points."""
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


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="PCA(N) + L2 + KMeans (+ ALWAYS: corpus term counts)")
    ap.add_argument("--model", required=True, help="Path to Gensim model or KeyedVectors")
    ap.add_argument("--vocab", required=True, help="Path to vocab (JSON or TXT)")
    ap.add_argument("--outdir", required=True, help="Directory for timestamped outputs")

    ap.add_argument("--pca-dim", type=int, default=46, help="PCA target dimensionality before clustering (default 46)")
    ap.add_argument("--k-min", type=int, default=3, help="Minimum K to try")
    ap.add_argument("--k-max", type=int, default=30, help="Maximum K to try")
    ap.add_argument("--n-init", type=int, default=20, help="KMeans n_init (default 20)")
    ap.add_argument("--max-iter", type=int, default=500, help="KMeans max_iter (default 500)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--top-terms", type=int, default=25, help="Top terms per cluster in summary (default 25)")

    # Docs source: n-gram phrasing optional; fallback to processed docs without n-grams
    ap.add_argument("--ngram-phraser-dir", default=None,
                    help="Directory containing trained n-gram phrasers; if omitted, fall back to process_all_noburp() (no n-grams).")
    ap.add_argument("--top-freq", type=int, default=50,
                    help="Top-N terms by frequency per cluster (default 50)")
    ap.add_argument("--min-sim", type=float, default=0.0,
                    help="Minimum sim_to_centroid to include a term (default 0.0)")

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

    # PCA -> D dims
    requested_d = int(args.pca_dim)
    max_d = int(min(len(in_vocab), X.shape[1]))
    if requested_d > max_d:
        logging.warning(f"--pca-dim {requested_d} > max feasible {max_d}; using {max_d} instead.")
    D = int(min(requested_d, max_d))
    logging.info(f"Fitting PCA(n_components={D}) on subset")
    pca = PCA(n_components=D, svd_solver="auto", random_state=args.random_state)
    Z = pca.fit_transform(X)
    var_expl = float(np.sum(pca.explained_variance_ratio_))
    logging.info(f"PCA variance explained (@{D}): {var_expl:.4f}")

    # L2-normalize
    Z = normalize(Z, norm="l2", axis=1)

    # Persist PCA transformer and the D-dim normalized embedding
    dump(pca, outdir / f"pca_{D}.joblib")
    np.savez_compressed(outdir / f"Z_{D}_l2.npz", tokens=np.array(in_vocab, dtype=object), Z=Z)

    # K sweep
    Ks = np.arange(int(args.k_min), int(args.k_max) + 1)
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
        idx_knee = knee_max_distance(Ks_arr, inertia_arr)
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
        if np.sum(mask & (Ks_arr == K_final)) > 1:
            K_final = int(Ks_arr[np.nanargmax(np.where(mask, comp, -np.inf))])
    else:
        K_final = K_comp if not np.isnan(K_comp) else K_sil

    # Plots
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
    # ALWAYS: Per-cluster term frequency over corpus
    # -------------------------
    if args.ngram_phraser_dir:
        logging.info(f"Loading tokenized+phrased docs with process_ngram_docs(ngram_phraser_dir='{args.ngram_phraser_dir}')")
        docs_iter = process_ngram_docs(ngram_phraser_dir=args.ngram_phraser_dir)
        docs_source = "ngram_phrased"
    else:
        logging.info("No --ngram-phraser-dir provided; falling back to process_all_noburp() (no n-gram phrasing).")
        # Use defaults that include lemmatization + stopword removal
        docs_iter = process_all_noburp(show_progress=True)
        docs_source = "processed_no_ngrams"

    # map term -> (cluster, sim_to_centroid)
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

    logging.info(f"Counted cluster-term frequencies over {total_docs:,} docs "
                 f"({total_tokens_matched:,} matched tokens).")

    # Build JSON with top-N by frequency per cluster, filtered by min similarity
    top_n = int(args.top_freq)
    min_sim = float(args.min_sim)
    clusters_out = []
    for k in range(km_best.n_clusters):
        cnt = per_cluster_counts.get(k, Counter())
        # filter by similarity threshold
        filtered_items = [(t, c) for t, c in cnt.items() if term2sim.get(t, -1.0) >= min_sim]
        if not filtered_items:
            logging.warning(f"Cluster {k}: 0 terms meet min-sim={min_sim:.3f}; output will be empty for this cluster.")
        # sort by (-count, -sim, term)
        filtered_items.sort(key=lambda kv: (-kv[1], -term2sim.get(kv[0], 0.0), kv[0]))

        # structured list
        top_struct = [
            {"term": t, "count": int(c), "sim_to_centroid": round(term2sim.get(t, 0.0), 6)}
            for t, c in filtered_items[:top_n]
        ]
        # newline-formatted list for readability
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
        "pca_components": int(D),
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
    for k in range(km_best.n_clusters):
        idx_k = np.where((lab_best == k) & (sims >= min_sim))[0]
        order = np.argsort(-sims[idx_k])
        idx_k = idx_k[order]
        terms_above = [in_vocab[i] for i in idx_k]
        clusters_terms_list[str(k)] = terms_above
        clusters_terms_lines[str(k)] = "\n".join(terms_above)

    terms_min_sim_list_path = outdir / f"cluster_terms_minSim_K{K_final}.json"
    terms_min_sim_list_path.write_text(json.dumps(clusters_terms_list, indent=2), encoding="utf-8")
    logging.info(f"Wrote cluster→terms (list) JSON to {terms_min_sim_list_path}")

    terms_min_sim_lines_path = outdir / f"cluster_terms_minSim_lines_K{K_final}.json"
    terms_min_sim_lines_path.write_text(json.dumps(clusters_terms_lines, indent=2), encoding="utf-8")
    logging.info(f"Wrote cluster→terms (newline) JSON to {terms_min_sim_lines_path}")

    # Decision file
    selection = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "embedding_dim": int(emb_dim),
        "n_terms": int(len(in_vocab)),
        "pca_components": int(D),
        "pca_variance_explained": float(var_expl),
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
        "docs_source": docs_source,
        "ngram_phraser_dir": args.ngram_phraser_dir,
        "min_sim": min_sim,
        "top_n": top_n,
    }
    (outdir / "selection_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    logging.info(
        f"Chosen K = {selection['K_final']} | sil={selection['metrics_at_final']['silhouette']:.4f} "
        f"| inertia={selection['metrics_at_final']['inertia']:.2f} | cohesion={selection['metrics_at_final']['cohesion_mean']:.4f} "
        f"| PCA D={D} (var_expl={var_expl:.4f}) | docs_source={docs_source} | min_sim={min_sim:.3f}"
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
