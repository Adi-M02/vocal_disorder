#!/usr/bin/env python3
"""
UMAP (dims ∈ {50,30,25,10}) + HDBSCAN grid on Word2Vec vocab, with metrics and top terms.

Requirements:
  pip install umap-learn hdbscan scikit-learn gensim matplotlib joblib pandas

Pipeline per dimension d ∈ {50,30,25,10}:
  1) Load Word2Vec/KeyedVectors and the input vocabulary (JSON or TXT).
  2) Build X for in-vocab tokens (n x 300 or model.dim).
  3) UMAP(n_components=d, n_neighbors=30, min_dist=0.1, metric='cosine') → Z_d
  4) HDBSCAN grid (metric='euclidean' on Z_d):
        - min_cluster_size ∈ {10, 20, 40, 80}
        - min_samples ∈ {None, =min_cluster_size}  (select via --min-samples-mode)
     cluster_selection_method='eom'.
  5) For each setting: compute metrics on non-noise points
        - n_clusters, noise_frac, silhouette, cluster persistence stats
        - borderline fractions (<0.30, <0.50), mean membership prob, size imbalance
        - composite score used to pick best setting for this dimension
  6) Save metrics CSV & plots. Save best assignments and top-50 terms/cluster.

Outputs (timestamped dir):
  - tokens_used.txt, oov_terms.txt
  - umap_Z_<d>.npz  (tokens, Z) for each dimension
  - umap_hdbscan_metrics.csv (all dims & settings)
  - plots/  (per-dim: noise_vs_mcs_<d>.png, silhouette_vs_mcs_<d>.png, persistence_vs_mcs_<d>.png)
  - clusters_<d>_<setting>.csv  (token, label, prob, sim_to_centroid)
  - cluster_summaries_<d>_<setting>.json
  - selection_summary.json  (winner per dimension)

Usage:
  python umap_hdbscan_grid.py \
      --model path/to/model.model \
      --vocab path/to/terms.json \
      --outdir runs/umap_hdb \
      [--dims 50 30 25 10] \
      [--umap-nn 30 --umap-min-dist 0.1] \
      [--mcs 10 20 40 80] \
      [--min-samples-mode both] \
      [--top-terms 50] \
      [--random-state 42]
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump

try:
    import umap
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires 'umap-learn'. Install via: pip install umap-learn") from e

try:
    import hdbscan
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires 'hdbscan'. Install via: pip install hdbscan") from e

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

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
    (run_dir / 'plots').mkdir(parents=True, exist_ok=True)
    return run_dir


def load_vocab(path: Path) -> List[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("seed_terms", "terms", "vocabulary"):
            if key in data and isinstance(data[key], list):
                return [str(t) for t in data[key]]
        raise ValueError(f"JSON {path} missing one of keys: seed_terms | terms | vocabulary")
    # TXT
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            terms.append(line)
    if not terms:
        raise ValueError(f"No terms found in text file: {path}")
    return terms


def load_w2v_any(path: Path):
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


def setting_id(min_cluster_size: int, min_samples: Optional[int]) -> str:
    return f"mcs{min_cluster_size}_ms{min_samples if min_samples is not None else 'None'}"


def cluster_centroids(Z: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    for k in np.unique(labels):
        if k == -1:
            continue
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        c = Z[idx].mean(axis=0)
        nrm = float(np.linalg.norm(c))
        if nrm > 0:
            c = c / nrm
        cents[int(k)] = c
    return cents


def summarize_clusters(tokens: List[str], Z: np.ndarray, labels: np.ndarray, probs: np.ndarray, top_terms: int = 50) -> List[dict]:
    # L2-normalize Z for cosine-like sims to centroids (does not affect saved Z)
    Z_n = normalize(Z, norm='l2', axis=1)
    cents = cluster_centroids(Z_n, labels)
    summaries: List[dict] = []
    for k, c in sorted(cents.items()):
        idx = np.where(labels == k)[0]
        sims = (Z_n[idx] @ c)
        order = np.argsort(-sims)[: top_terms]
        top = [tokens[i] for i in idx[order]]
        mean_sim = float(np.mean(sims)) if idx.size else 0.0
        mean_prob = float(np.mean(probs[idx])) if idx.size else 0.0
        summaries.append({
            'cluster': int(k),
            'size': int(idx.size),
            'mean_sim': mean_sim,
            'mean_membership_prob': mean_prob,
            'top_terms': top,
        })
    return summaries


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description='UMAP + HDBSCAN grid for Word2Vec vocab')
    ap.add_argument('--model', required=True, help='Path to Gensim model or KeyedVectors')
    ap.add_argument('--vocab', required=True, help='Path to vocab (JSON or TXT)')
    ap.add_argument('--outdir', required=True, help='Directory for timestamped outputs')

    ap.add_argument('--dims', type=int, nargs='+', default=[50, 30, 25, 10], help='UMAP dimensions to try (default: 50 30 25 10)')
    ap.add_argument('--umap-nn', type=int, default=30, help='UMAP n_neighbors (default 30)')
    ap.add_argument('--umap-min-dist', type=float, default=0.1, help='UMAP min_dist (default 0.1)')

    ap.add_argument('--mcs', type=int, nargs='+', default=[10, 20, 40, 80], help='HDBSCAN min_cluster_size values')
    ap.add_argument('--min-samples-mode', choices=['both', 'none', 'equal'], default='both',
                    help="How to set min_samples: 'none'->None; 'equal'->=mcs; 'both'->try both")

    ap.add_argument('--top-terms', type=int, default=50, help='Top terms per cluster in summary (default 50)')
    ap.add_argument('--random-state', type=int, default=42, help='Random seed (default 42)')

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    model_path = Path(args.model)
    vocab_path = Path(args.vocab)
    outdir = ensure_outdir(Path(args.outdir))
    (outdir / 'plots').mkdir(exist_ok=True)
    logging.info(f'Output directory: {outdir}')

    # Load model
    logging.info(f'Loading model: {model_path}')
    kv = load_w2v_any(model_path)
    emb_dim = int(kv.vector_size)
    logging.info(f'Model vocab size: {len(kv.key_to_index):,} | dim: {emb_dim}')

    # Load vocab
    terms = load_vocab(vocab_path)
    logging.info(f'Loaded {len(terms):,} terms from {vocab_path}')

    # Intersect with model vocab
    in_vocab = [t for t in terms if t in kv]
    oov = [t for t in terms if t not in kv]
    if not in_vocab:
        raise SystemExit('None of the provided terms are in the model vocabulary.')
    logging.info(f'In-vocab terms: {len(in_vocab):,} | OOV terms: {len(oov):,}')

    # Save OOV & used tokens
    (outdir / 'tokens_used.txt').write_text('\n'.join(in_vocab), encoding='utf-8')
    if oov:
        (outdir / 'oov_terms.txt').write_text('\n'.join(oov), encoding='utf-8')

    # Build matrix for subset
    X = kv[in_vocab]  # shape: (n_tokens, emb_dim)

    # Metrics CSV header
    import csv
    metrics_path = outdir / 'umap_hdbscan_metrics.csv'
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'dim','setting','min_cluster_size','min_samples','n_clusters','noise_frac',
            'silhouette_nonnoise','persistence_mean','persistence_median','persistence_p10','persistence_p90',
            'borderline_frac_lt0.30','borderline_frac_lt0.50','mean_prob_nonnoise','size_imbalance','composite_score'
        ])

    # Prepare HDBSCAN settings
    mcs_values = sorted(set(int(x) for x in args.mcs))
    ms_modes = [args.min_samples_mode] if args.min_samples_mode != 'both' else ['none','equal']
    settings: List[Tuple[int, Optional[int]]] = []
    for mcs in mcs_values:
        for mode in ms_modes:
            ms = None if mode == 'none' else int(mcs)
            settings.append((mcs, ms))

    selections = {}

    for dim in args.dims:
        logging.info(f'UMAP: dim={dim}, n_neighbors={args.umap_nn}, min_dist={args.umap_min_dist}')
        reducer = umap.UMAP(n_components=int(dim), n_neighbors=int(args.umap_nn), min_dist=float(args.umap_min_dist),
                            metric='cosine', random_state=args.random_state)
        Z = reducer.fit_transform(X)
        np.savez_compressed(outdir / f'umap_Z_{dim}.npz', tokens=np.array(in_vocab, dtype=object), Z=Z)

        best_score = -np.inf
        best_result = None  # (labels, probs, setting_str)
        metrics_rows_dim = []

        for mcs, ms in settings:
            sett = setting_id(mcs, ms)
            logging.info(f'  HDBSCAN on dim={dim} | setting={sett}')
            clusterer = hdbscan.HDBSCAN(min_cluster_size=int(mcs),
                                        min_samples=None if ms is None else int(ms),
                                        metric='euclidean',
                                        cluster_selection_method='eom',
                                        prediction_data=False,
                                        approx_min_span_tree=True)
            labels = clusterer.fit_predict(Z)
            probs = clusterer.probabilities_

            # Basic counts/masks
            n = Z.shape[0]
            mask_nn = labels != -1
            n_nn = int(np.sum(mask_nn))
            n_clusters = int(len(set(labels[mask_nn])))
            noise_frac = float(1.0 - (n_nn / float(n)))

            # Silhouette on non-noise
            if n_clusters >= 2 and n_nn >= 10:
                try:
                    sil = float(silhouette_score(Z[mask_nn], labels[mask_nn], metric='euclidean'))
                except Exception:
                    sil = float('nan')
            else:
                sil = float('nan')

            # Persistence
            persistence = getattr(clusterer, 'cluster_persistence_', None)
            persistence_by_label: Dict[int, float] = {}
            if persistence is not None and len(persistence) > 0:
                uniq_sorted = sorted([lab for lab in np.unique(labels) if lab != -1])
                if len(uniq_sorted) == len(persistence):
                    for lab, pers in zip(uniq_sorted, persistence):
                        persistence_by_label[int(lab)] = float(pers)
                else:
                    try:
                        pers_dict = clusterer.condensed_tree_.cluster_persistence()
                        for lab in uniq_sorted:
                            persistence_by_label[int(lab)] = float(pers_dict.get(int(lab), np.nan))
                    except Exception:
                        pass
            if not persistence_by_label:
                for lab in np.unique(labels):
                    if lab != -1:
                        persistence_by_label[int(lab)] = float('nan')

            pers_vals = np.array([persistence_by_label.get(int(k), np.nan) for k in sorted(set(labels) - {-1})], dtype=float)
            pers_mean = float(np.nanmean(pers_vals)) if pers_vals.size else float('nan')
            pers_median = float(np.nanmedian(pers_vals)) if pers_vals.size else float('nan')
            p10 = float(np.nanpercentile(pers_vals, 10)) if pers_vals.size else float('nan')
            p90 = float(np.nanpercentile(pers_vals, 90)) if pers_vals.size else float('nan')
            borderline_03 = float(np.mean(pers_vals < 0.30)) if pers_vals.size else float('nan')
            borderline_05 = float(np.mean(pers_vals < 0.50)) if pers_vals.size else float('nan')

            mean_prob_nn = float(np.mean(probs[mask_nn])) if n_nn > 0 else float('nan')

            # Size imbalance on non-noise labels
            if n_clusters > 0:
                labels_nn = labels[mask_nn].astype(int, copy=False)
                sizes = np.bincount(labels_nn)
                size_imb = float(np.std(sizes) / (np.mean(sizes) + 1e-12))
            else:
                size_imb = float('nan')

            # Composite score
            noise_score = 1.0 - noise_frac
            sil_score = 0.0 if np.isnan(sil) else max(0.0, sil)
            border_score = 1.0 - (borderline_03 if not np.isnan(borderline_03) else 1.0)
            pers_score = 0.0 if np.isnan(pers_median) else pers_median
            comp = 0.4*noise_score + 0.3*border_score + 0.2*pers_score + 0.1*sil_score

            metrics_rows_dim.append([
                int(dim), sett, mcs, ('' if ms is None else ms), n_clusters, noise_frac,
                sil, pers_mean, pers_median, p10, p90, borderline_03, borderline_05,
                mean_prob_nn, size_imb, comp
            ])

            if comp > best_score:
                best_score = comp
                best_result = (labels, probs, sett)

            logging.info(
                f"    {sett} | clusters={n_clusters} | noise={noise_frac:.3f} | sil={sil:.4f} | "
                f"pers_med={pers_median:.3f} | comp={comp:.3f}"
            )

        # Append metrics for this dim
        with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for row in metrics_rows_dim:
                w.writerow([
                    row[0], row[1], int(row[2]), row[3], int(row[4]), f"{row[5]:.6f}",
                    f"{row[6]:.6f}" if not np.isnan(row[6]) else 'nan',
                    f"{row[7]:.6f}" if not np.isnan(row[7]) else 'nan',
                    f"{row[8]:.6f}" if not np.isnan(row[8]) else 'nan',
                    f"{row[9]:.6f}" if not np.isnan(row[9]) else 'nan',
                    f"{row[10]:.6f}" if not np.isnan(row[10]) else 'nan',
                    f"{row[11]:.6f}" if not np.isnan(row[11]) else 'nan',
                    f"{row[12]:.6f}" if not np.isnan(row[12]) else 'nan',
                    f"{row[13]:.6f}" if not np.isnan(row[13]) else 'nan',
                    f"{row[14]:.6f}"
                ])

        # Plots for this dim
        try:
            import pandas as pd
            df = pd.DataFrame(metrics_rows_dim, columns=[
                'dim','setting','mcs','ms','n_clusters','noise','sil','pers_mean','pers_med','p10','p90',
                'border03','border05','mean_prob','size_imb','comp'
            ])
            df['ms_label'] = df['ms'].apply(lambda v: 'None' if v == '' else ('=mcs' if str(v).isdigit() else str(v)))

            # Noise
            plt.figure(figsize=(7,4))
            for lbl, sub in df.groupby('ms_label'):
                x = sub['mcs'].astype(int).values
                y = sub['noise'].astype(float).values
                idx = np.argsort(x)
                plt.plot(x[idx], y[idx], marker='o', linewidth=1, label=f'min_samples {lbl}')
            plt.xlabel('min_cluster_size')
            plt.ylabel('Noise fraction')
            plt.title(f'HDBSCAN noise vs setting (dim={dim})')
            plt.legend(); plt.tight_layout()
            plt.savefig(outdir / 'plots' / f'noise_vs_mcs_{dim}.png', dpi=160)
            plt.close()

            # Silhouette
            plt.figure(figsize=(7,4))
            for lbl, sub in df.groupby('ms_label'):
                x = sub['mcs'].astype(int).values
                y = sub['sil'].astype(float).values
                idx = np.argsort(x)
                plt.plot(x[idx], y[idx], marker='o', linewidth=1, label=f'min_samples {lbl}')
            plt.xlabel('min_cluster_size')
            plt.ylabel('Silhouette (non-noise)')
            plt.title(f'HDBSCAN silhouette vs setting (dim={dim})')
            plt.legend(); plt.tight_layout()
            plt.savefig(outdir / 'plots' / f'silhouette_vs_mcs_{dim}.png', dpi=160)
            plt.close()

            # Persistence median
            plt.figure(figsize=(7,4))
            for lbl, sub in df.groupby('ms_label'):
                x = sub['mcs'].astype(int).values
                y = sub['pers_med'].astype(float).values
                idx = np.argsort(x)
                plt.plot(x[idx], y[idx], marker='o', linewidth=1, label=f'min_samples {lbl}')
            plt.xlabel('min_cluster_size')
            plt.ylabel('Median cluster persistence')
            plt.title(f'HDBSCAN persistence vs setting (dim={dim})')
            plt.legend(); plt.tight_layout()
            plt.savefig(outdir / 'plots' / f'persistence_vs_mcs_{dim}.png', dpi=160)
            plt.close()
        except Exception:
            pass

        # Save best assignments & summaries for this dim
        if best_result is None:
            raise SystemExit('HDBSCAN produced no result; check inputs.')
        labels, probs, best_setting = best_result

        # Assignments
        with open(outdir / f'clusters_{dim}_{best_setting}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['token','label','prob','sim_to_centroid'])
            cents = cluster_centroids(normalize(Z, norm='l2', axis=1), labels)
            for i, tok in enumerate(in_vocab):
                lab = int(labels[i])
                prob = float(probs[i])
                sim = float(normalize(Z[i:i+1], norm='l2', axis=1) @ cents[lab]) if (lab != -1 and lab in cents) else float('nan')
                w.writerow([tok, lab, f'{prob:.6f}', f'{sim:.6f}' if not np.isnan(sim) else 'nan'])

        # Summaries
        summaries = summarize_clusters(in_vocab, Z, labels, probs, top_terms=int(args.top_terms))
        (outdir / f'cluster_summaries_{dim}_{best_setting}.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')

        selections[int(dim)] = {
            'best_setting': best_setting,
            'n_clusters': int(len(set(labels[labels != -1]))),
            'noise_frac': float(np.mean(labels == -1)),
        }

    # Selection summary across dims
    (outdir / 'selection_summary.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')

    logging.info('Done.')


if __name__ == '__main__':
    main()
