#!/usr/bin/env python3
"""
Cluster Word2Vec terms with PCA (to 46D) + L2 normalization + HDBSCAN grid.

Requirements:
  pip install hdbscan scikit-learn gensim matplotlib joblib

Inputs:
  - Gensim Word2Vec/KeyedVectors model
  - Vocab JSON/TXT (JSON supports keys: {"seed_terms"|"terms"|"vocabulary"})

Pipeline:
  1) Load model + intersect with provided vocab
  2) Fit PCA (n_components=46) on the *subset* vectors and transform them
  3) L2-normalize rows (places points on unit sphere)
  4) Grid-search HDBSCAN with selectable metric:
        - metric='euclidean' (default; with L2 data ≈ cosine)
        - metric='cosine_precomputed' (exact cosine via NxN distance matrix)
        - metric='cosine' (alias for precomputed cosine)
        min_cluster_size ∈ {10, 20, 40, 80}
        min_samples ∈ {None, equal to min_cluster_size}
     Use cluster_selection_method='eom'.
  5) For each run, compute:
       - n_clusters (labels != -1)
       - noise_frac (labels == -1 proportion)
       - silhouette over non-noise points (euclidean on unit vectors ~ cosine, or precomputed cosine distance)
       - cluster persistence stats (mean/median/p10/p90)
       - borderline cluster fractions (persistence < 0.30 and < 0.50)
       - mean membership probability for non-noise points
       - size_imbalance over non-noise clusters: std(size)/mean(size)
       - composite score to pick the best run
  6) Save cluster assignments and per-cluster top terms near centroid

Outputs (timestamped directory):
  - tokens_used.txt, oov_terms.txt
  - pca.joblib (fitted PCA(46))
  - Z_46_l2.npz  (tokens + 46D normalized embeddings)
  - hdbscan_grid_metrics.csv  (one row per setting)
  - noise_vs_setting.png, silhouette_vs_setting.png, persistence_vs_setting.png
  - selection_summary.json (chosen setting + metrics)
  - clusters_<setting>.csv  (token, label, prob, sim_to_centroid)
  - cluster_summaries_<setting>.json (per cluster: size, mean_sim, persistence, top_terms)

Usage:
  python cluster_pca_hdbscan.py \
      --model path/to/model.model \
      --vocab path/to/terms.json \
      --outdir runs/cluster_hdb \
      [--mcs 10 20 40 80] \
      [--min-samples-mode both] \
      [--metric euclidean|cosine_precomputed] \
      [--random-state 42] \
      [--top-terms 25]

Notes:
  - PCA is fit on the provided subset by default; adapt to fit on kv.vectors if desired.
  - With unit vectors, euclidean geometry approximates cosine behavior; for exact cosine use --metric cosine_precomputed.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from joblib import dump
import matplotlib.pyplot as plt

try:
    import hdbscan
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires the 'hdbscan' package. Install it via 'pip install hdbscan'.") from e

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


def setting_id(min_cluster_size: int, min_samples: Optional[int]) -> str:
    return f"mcs{min_cluster_size}_ms{min_samples if min_samples is not None else 'None'}"


def cluster_centroids(Z: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    centroids = {}
    for k in np.unique(labels):
        if k == -1:
            continue
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        c = Z[idx].mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-12)
        centroids[int(k)] = c
    return centroids


def summarize_clusters(tokens: List[str], Z: np.ndarray, labels: np.ndarray, probs: np.ndarray, persistence_by_label: Dict[int, float], top_terms: int = 25) -> List[dict]:
    cents = cluster_centroids(Z, labels)
    out = []
    for k, c in sorted(cents.items()):
        idx = np.where(labels == k)[0]
        sims = (Z[idx] @ c)
        order = np.argsort(-sims)[: top_terms]
        top = [tokens[i] for i in idx[order]]
        mean_sim = float(np.mean(sims)) if idx.size else 0.0
        mean_prob = float(np.mean(probs[idx])) if idx.size else 0.0
        out.append({
            "cluster": int(k),
            "size": int(idx.size),
            "mean_sim": mean_sim,
            "mean_membership_prob": mean_prob,
            "persistence": float(persistence_by_label.get(int(k), float('nan'))),
            "top_terms": top,
        })
    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="PCA(46) + L2 + HDBSCAN grid for Word2Vec vocab")
    ap.add_argument("--model", required=True, help="Path to Gensim model or KeyedVectors")
    ap.add_argument("--vocab", required=True, help="Path to vocab (JSON or TXT)")
    ap.add_argument("--outdir", required=True, help="Directory for timestamped outputs")

    ap.add_argument("--mcs", type=int, nargs='+', default=[10, 20, 40, 80], help="Values for min_cluster_size (default: 10 20 40 80)")
    ap.add_argument("--min-samples-mode", choices=["both", "none", "equal"], default="both",
                    help="How to set min_samples: 'none' -> None; 'equal' -> =min_cluster_size; 'both' -> try both")
    ap.add_argument("--metric", choices=["euclidean", "cosine", "cosine_precomputed"], default="euclidean",
                    help="Distance metric for HDBSCAN. 'euclidean' on L2-normalized data approximates cosine; use 'cosine_precomputed' to force exact cosine via an NxN distance matrix.")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for PCA (default 42)")
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
    (outdir / "tokens_used.txt").write_text("".join(in_vocab), encoding="utf-8")
    if oov:
        (outdir / "oov_terms.txt").write_text("".join(oov), encoding="utf-8")

    # Build matrix for subset
    X = kv[in_vocab]  # shape: (n_tokens, emb_dim)

    # PCA -> 46 dims
    logging.info("Fitting PCA(n_components=46) on subset")
    pca = PCA(n_components=46, svd_solver="auto", random_state=args.random_state)
    Z = pca.fit_transform(X)

    # L2-normalize
    Z = normalize(Z, norm="l2", axis=1)

    # Decide metric handling
    metric_mode = args.metric
    logging.info(f"Clustering metric mode: {metric_mode}")

    # If using cosine precomputed, build the NxN distance matrix once (float32 to save memory)
    D = None
    if metric_mode in ("cosine", "cosine_precomputed"):
        n = Z.shape[0]
        logging.info(f"Computing cosine distance matrix of shape {n}x{n} (float32)")
        # cosine distance = 1 - dot(u, v) for unit vectors
        # Use blocks if needed for memory; for typical n~7k this fits in RAM (~200MB float32)
        Zf = Z.astype(np.float32, copy=False)
        D = 1.0 - np.clip(Zf @ Zf.T, -1.0, 1.0)
        np.fill_diagonal(D, 0.0)
        metric_param = 'precomputed'
    else:
        metric_param = 'euclidean'

    # Persist PCA transformer and the 46D normalized embedding
    dump(pca, outdir / "pca.joblib")
    np.savez_compressed(outdir / "Z_46_l2.npz", tokens=np.array(in_vocab, dtype=object), Z=Z)

    # Build parameter grid
    mcs_values = list(sorted(set(int(x) for x in args.mcs)))
    ms_modes = [args.min_samples_mode]
    if args.min_samples_mode == "both":
        ms_modes = ["none", "equal"]

    settings: List[Tuple[int, Optional[int]]] = []
    for mcs in mcs_values:
        for mode in ms_modes:
            if mode == "none":
                ms = None
            elif mode == "equal":
                ms = int(mcs)
            else:
                raise ValueError("Unexpected min-samples-mode")
            settings.append((mcs, ms))

    # Metrics storage
    import csv
    metrics_rows = []

    best_score = -np.inf
    best_result = None  # (labels, probs, persistence_by_label, setting_str, clusterer)

    # Evaluate each setting
    for mcs, ms in settings:
        sett = setting_id(mcs, ms)
        logging.info(f"HDBSCAN setting: {sett}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=None if ms is None else int(ms),
            metric=metric_param,
            cluster_selection_method='eom',
            prediction_data=False,
            approx_min_span_tree=True
        )
        if metric_param == 'precomputed':
            labels = clusterer.fit_predict(D)
        else:
            labels = clusterer.fit_predict(Z)
        probs = clusterer.probabilities_

        # Basic counts/masks
        n = Z.shape[0]
        mask_nn = labels != -1
        n_nn = int(np.sum(mask_nn))
        n_clusters = int(len(set(labels[mask_nn])))
        noise_frac = float(1.0 - (n_nn / float(n)))

        # Cluster persistence per label
        persistence = getattr(clusterer, 'cluster_persistence_', None)
        persistence_by_label: Dict[int, float] = {}
        if persistence is not None and len(persistence) > 0:
            # hdbscan aligns cluster_persistence_ with labels 0..n_clusters-1
            uniq = [lab for lab in np.unique(labels) if lab != -1]
            uniq_sorted = sorted(uniq)
            if len(uniq_sorted) == len(persistence):
                for lab, pers in zip(uniq_sorted, persistence):
                    persistence_by_label[int(lab)] = float(pers)
            else:
                # Fallback via condensed_tree if misaligned
                try:
                    _df = clusterer.condensed_tree_.to_pandas()
                    pers_dict = clusterer.condensed_tree_.cluster_persistence()
                    for lab in uniq_sorted:
                        persistence_by_label[int(lab)] = float(pers_dict.get(int(lab), np.nan))
                except Exception:
                    pass
        # If still empty, fill NaNs
        if not persistence_by_label:
            for lab in np.unique(labels):
                if lab != -1:
                    persistence_by_label[int(lab)] = float('nan')

        # Compute silhouette on non-noise
        if n_clusters >= 2 and n_nn >= 10:
            if metric_param == 'precomputed':
                idx = np.where(mask_nn)[0]
                D_nn = D[np.ix_(idx, idx)]
                sil = float(silhouette_score(D_nn, labels[mask_nn], metric='precomputed'))
            else:
                sil = float(silhouette_score(Z[mask_nn], labels[mask_nn], metric='euclidean'))
        else:
            sil = float('nan')

        # Persistence stats on non-noise clusters
        pers_vals = np.array([persistence_by_label.get(int(k), np.nan) for k in sorted(set(labels) - {-1})], dtype=float)
        pers_mean = float(np.nanmean(pers_vals)) if pers_vals.size else float('nan')
        pers_median = float(np.nanmedian(pers_vals)) if pers_vals.size else float('nan')
        p10 = float(np.nanpercentile(pers_vals, 10)) if pers_vals.size else float('nan')
        p90 = float(np.nanpercentile(pers_vals, 90)) if pers_vals.size else float('nan')
        borderline_03 = float(np.mean(pers_vals < 0.30)) if pers_vals.size else float('nan')
        borderline_05 = float(np.mean(pers_vals < 0.50)) if pers_vals.size else float('nan')

        mean_prob_nn = float(np.mean(probs[mask_nn])) if n_nn > 0 else float('nan')

        # Cluster size imbalance among non-noise clusters
        if n_clusters > 0:
            labels_nn = labels[mask_nn].astype(int, copy=False)
            sizes = np.bincount(labels_nn)
            size_imb = float(np.std(sizes) / (np.mean(sizes) + 1e-12))
        else:
            size_imb = float('nan')

        # Composite selection score
        # Favor low noise, low borderline, high persistence median, decent silhouette
        noise_score = 1.0 - noise_frac
        sil_score = 0.0 if np.isnan(sil) else max(0.0, sil)
        border_score = 1.0 - (borderline_03 if not np.isnan(borderline_03) else 1.0)
        pers_score = 0.0 if np.isnan(pers_median) else pers_median
        comp = 0.4*noise_score + 0.3*border_score + 0.2*pers_score + 0.1*sil_score

        metrics_rows.append([
            setting_id(mcs, ms), mcs, ('' if ms is None else ms), n_clusters, noise_frac,
            sil, pers_mean, pers_median, p10, p90, borderline_03, borderline_05,
            mean_prob_nn, size_imb, comp
        ])

        # Keep best
        if comp > best_score:
            best_score = comp
            best_result = (labels, probs, persistence_by_label, setting_id(mcs, ms), clusterer)

        logging.info(
            f"{sett} | clusters={n_clusters} | noise={noise_frac:.3f} | sil={sil:.4f} | "
            f"pers_med={pers_median:.3f} | borderline<0.3={borderline_03:.2f} | comp={comp:.3f}"
        )

    # Save metrics CSV
    import csv
    with open(outdir / "hdbscan_grid_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "setting","min_cluster_size","min_samples","n_clusters","noise_frac",
            "silhouette_nonnoise","persistence_mean","persistence_median","persistence_p10","persistence_p90",
            "borderline_frac_lt0.30","borderline_frac_lt0.50","mean_prob_nonnoise","size_imbalance","composite_score"
        ])
        for row in metrics_rows:
            w.writerow([
                row[0], int(row[1]), row[2], int(row[3]), f"{row[4]:.6f}",
                f"{row[5]:.6f}" if not np.isnan(row[5]) else "nan",
                f"{row[6]:.6f}" if not np.isnan(row[6]) else "nan",
                f"{row[7]:.6f}" if not np.isnan(row[7]) else "nan",
                f"{row[8]:.6f}" if not np.isnan(row[8]) else "nan",
                f"{row[9]:.6f}" if not np.isnan(row[9]) else "nan",
                f"{row[10]:.6f}" if not np.isnan(row[10]) else "nan",
                f"{row[11]:.6f}" if not np.isnan(row[11]) else "nan",
                f"{row[12]:.6f}" if not np.isnan(row[12]) else "nan",
                f"{row[13]:.6f}" if not np.isnan(row[13]) else "nan",
                f"{row[14]:.6f}"
            ])

    # Plots across settings (x-axis = min_cluster_size; lines for min_samples modes)
    try:
        import pandas as pd
        df = pd.DataFrame(metrics_rows, columns=[
            "setting","mcs","ms","n_clusters","noise","sil","pers_mean","pers_med","p10","p90",
            "border03","border05","mean_prob","size_imb","comp"
        ])
        # Normalize ms label for plotting
        df["ms_label"] = df["ms"].apply(lambda v: "None" if v == '' else ("=mcs" if str(v).isdigit() else str(v)))

        # Noise plot
        plt.figure(figsize=(7,4))
        for lbl, sub in df.groupby("ms_label"):
            x = sub["mcs"].astype(int).values
            y = sub["noise"].astype(float).values
            idx = np.argsort(x)
            plt.plot(x[idx], y[idx], marker='o', linewidth=1, label=f"min_samples {lbl}")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Noise fraction")
        ttl = "HDBSCAN noise vs setting"
        if metric_param == 'precomputed':
            ttl += " (cosine)"
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "noise_vs_setting.png", dpi=160)
        plt.close()

        # Silhouette plot
        plt.figure(figsize=(7,4))
        for lbl, sub in df.groupby("ms_label"):
            x = sub["mcs"].astype(int).values
            y = sub["sil"].astype(float).values
            idx = np.argsort(x)
            plt.plot(x[idx], y[idx], marker='o', linewidth=1, label=f"min_samples {lbl}")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Silhouette (non-noise)")
        ttl = "HDBSCAN silhouette vs setting"
        if metric_param == 'precomputed':
            ttl += " (cosine)"
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "silhouette_vs_setting.png", dpi=160)
        plt.close()

        # Persistence (median) plot
        plt.figure(figsize=(7,4))
        for lbl, sub in df.groupby("ms_label"):
            x = sub["mcs"].astype(int).values
            y = sub["pers_med"].astype(float).values
            idx = np.argsort(x)
            plt.plot(x[idx], y[idx], marker='o', linewidth=1, label=f"min_samples {lbl}")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Median cluster persistence")
        ttl = "HDBSCAN persistence vs setting"
        if metric_param == 'precomputed':
            ttl += " (cosine)"
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "persistence_vs_setting.png", dpi=160)
        plt.close()
    except Exception:
        # If pandas or plotting fails, skip plots gracefully
        pass

    # Best result: write assignments and summaries
    if best_result is None:
        raise SystemExit("HDBSCAN produced no result; check inputs.")

    labels, probs, persistence_by_label, best_setting, clusterer = best_result

    # Save assignments
    import csv
    with open(outdir / f"clusters_{best_setting}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "label", "prob", "sim_to_centroid"])
        cents = cluster_centroids(Z, labels)
        for i, tok in enumerate(in_vocab):
            lab = int(labels[i])
            prob = float(probs[i])
            sim = float(Z[i] @ cents[lab]) if (lab != -1 and lab in cents) else float('nan')
            w.writerow([tok, lab, f"{prob:.6f}", f"{sim:.6f}" if not np.isnan(sim) else "nan"])

    # Save cluster summaries
    summaries = summarize_clusters(in_vocab, Z, labels, probs, persistence_by_label, top_terms=int(args.top_terms))
    (outdir / f"cluster_summaries_{best_setting}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # Decision file
    selection = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "embedding_dim": int(emb_dim),
        "n_terms": int(len(in_vocab)),
        "pca_components": 46,
        "grid": {
            "min_cluster_size": mcs_values,
            "min_samples_mode": args.min_samples_mode,
            "metric_mode": metric_mode,
        },
        "best_setting": best_setting,
    }
    (outdir / "selection_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    logging.info(f"Chosen setting: {best_setting}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
