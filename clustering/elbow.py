#!/usr/bin/env python3
"""
PCA scree (per-component variance) chooser for Word2Vec embeddings — with
optional "all-but-the-top" (drop-top PCs) elbow detection and projection.

Given a Gensim Word2Vec (or KeyedVectors) model and an input vocabulary,
this script fits PCA, plots the **per-component explained variance** (scree),
detects an "elbow" (knee) on that curve, and saves:

- A timestamped output directory containing:
  - explained_variance.csv              # per-component variance stats + cumulative
  - scree_variance.png                  # scree plot with suggested k
  - decision.json                       # chosen k values and metadata
  - pca.joblib                          # fitted PCA transformer (max_components)
  - oov_terms.txt                       # terms not found in the model (if any)
  - tokens_used.txt                     # in-vocab tokens used for fitting
  - reduced_embeddings_elbow.npz        # projected coords @ k_elbow (after drop-top)
  - reduced_embeddings_target.npz       # projected coords @ k_target (after drop-top)

You can later load pca.joblib to transform any vectors from the SAME model space
into the reduced-dimension space.

Usage:
  python pca_elbow_from_w2v.py \
      --model path/to/model.model \
      --vocab path/to/terms.json \
      --outdir runs/pca \
      [--fit-on subset|model] \
      [--fit-topn 200000] \
      [--max-components 200] \
      [--target-variance 0.90] \
      [--method max_distance|curvature] \
      [--drop-top 0]

Vocab file formats supported:
  - JSON with one of these keys: {"seed_terms": [...]} or {"terms": [...]} or {"vocabulary": [...]}.
  - TXT with one term per line.

Notes:
  * "Elbow" detection is performed on the **scree** (per-component variance) curve.
  * If no clear elbow exists, we also compute the smallest k that meets
    --target-variance (cumulative; default 90%) and save that projection too.
  * The --drop-top option implements "all-but-the-top": drop the first r PCs
    (e.g., r=1..3) before elbow detection and when saving reduced embeddings.
  * By default, PCA is fit on your provided tokens (fit-on=subset). If you
    prefer a model-wide PCA basis, set --fit-on=model (optionally limited by
    --fit-topn most frequent tokens for memory/speed).
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from joblib import dump
import matplotlib.pyplot as plt

try:
    from gensim.models import Word2Vec, KeyedVectors
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires gensim to be installed.") from e


# -------------------------
# Utility: IO & parsing
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

    Assumes JSON with key 'seed_terms' containing a list of terms,
    or TXT with one term per line.
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "seed_terms" in data and isinstance(data["seed_terms"], list):
            print(str(t) for t in data["seed_terms"])
            return [str(t) for t in data["seed_terms"]]
        raise ValueError(f"JSON {path} missing key: seed_terms")
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


# -------------------------
# PCA & elbow detection helpers
# -------------------------

def explained_variance_table(pca: PCA) -> np.ndarray:
    """Return an (n_components x 3) table: [component_index, var_ratio, cum_var]."""
    vr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cum = np.cumsum(vr)
    idx = np.arange(1, len(vr) + 1)
    return np.column_stack([idx, vr, cum])


def knee_max_distance(y: np.ndarray) -> int:
    """Elbow by maximum distance from the line connecting first and last point.

    Works for decreasing scree curves as well as increasing cumulative curves.
    Returns 1-based index of suggested k.
    """
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
    return int(np.argmax(d)) + 1  # 1-based


def _smooth(y: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1 or win > len(y):
        return y.astype(float).copy()
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode='same')


def knee_curvature(y: np.ndarray) -> int:
    """Elbow by maximum discrete second derivative (curvature) on a smoothed scree.
    Returns 1-based index.
    """
    n = len(y)
    if n < 3:
        return n
    y_sm = _smooth(y, win=5)
    second = y_sm[:-2] - 2 * y_sm[1:-1] + y_sm[2:]
    return int(np.argmax(second)) + 2  # center index + 1 for 1-based


# -------------------------
# Main logic
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="PCA scree (per-component variance) chooser for Word2Vec embeddings")
    ap.add_argument("--model", required=True, help="Path to Gensim model or KeyedVectors")
    ap.add_argument("--vocab", required=True, help="Path to vocab (JSON or TXT)")
    ap.add_argument("--outdir", required=True, help="Directory for timestamped outputs")

    ap.add_argument("--fit-on", choices=["subset", "model"], default="subset",
                    help="Fit PCA on 'subset' (your tokens) or entire 'model' vectors")
    ap.add_argument("--fit-topn", type=int, default=200000,
                    help="If --fit-on=model, limit to top-N most frequent tokens (default 200k)")
    ap.add_argument("--max-components", type=int, default=60,
                    help="Maximum PCA components to compute (<= embedding dim)")
    ap.add_argument("--target-variance", type=float, default=0.90,
                    help="Fallback on cumulative variance: smallest k reaching this value (default 0.90)")
    ap.add_argument("--method", choices=["max_distance", "curvature"], default="max_distance",
                    help="Elbow detection method on the scree curve (default max_distance)")
    ap.add_argument("--drop-top", type=int, default=0,
                    help="Discard the first r principal components (all-but-the-top). 0 disables.")

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

    # Build matrices
    X_subset = kv[in_vocab]  # shape: (n_tokens, emb_dim)

    if args.fit_on == "subset":
        X_fit = X_subset
    else:
        # Fit on entire model (optionally limited by top-N tokens)
        logging.info("Fitting PCA on model-wide vectors")
        if args.fit_topn and args.fit_topn < len(kv.key_to_index):
            # key_to_index preserves descending frequency in Gensim 4
            top_tokens = list(kv.key_to_index.keys())[: args.fit_topn]
            X_fit = kv[top_tokens]
        else:
            X_fit = kv.vectors

    n_fit, emb_dim = X_fit.shape
    max_comps = min(args.max_components, emb_dim)

    drop_top = max(0, int(args.drop_top))
    if max_comps <= 0:
        raise SystemExit("max_components must be >= 1")
    if drop_top >= max_comps:
        logging.warning("--drop-top >= max_components; clamping to max_components-1")
        drop_top = max_comps - 1

    logging.info(
        f"Fitting PCA on shape: {X_fit.shape} | max_components={max_comps} | drop_top={drop_top}"
    )

    # Fit PCA (randomized solver is fast for many comps)
    pca = PCA(n_components=max_comps, svd_solver="randomized", random_state=42)
    pca.fit(X_fit)

    # Variance tables
    table = explained_variance_table(pca)  # columns: [idx, var_ratio, cum_var]
    np.savetxt(
        outdir / "explained_variance.csv",
        table,
        delimiter=",",
        fmt=["%d", "%.8f", "%.8f"],
        header="component,var_ratio,cum_var",
        comments="",
    )

    var_ratio = table[:, 1]
    cum = table[:, 2]

    # Apply all-but-the-top for knee search and residual cumulative
    var_for_knee = var_ratio[drop_top:]
    if var_for_knee.size == 0:
        raise SystemExit("After dropping top PCs, no components remain for elbow detection.")

    # Elbow on scree (after dropping top PCs)
    if args.method == "max_distance":
        k_eff_elbow = knee_max_distance(var_for_knee)
    else:
        k_eff_elbow = knee_curvature(var_for_knee)
    k_elbow = drop_top + k_eff_elbow  # index in the full list (1-based in plots/logs)

    # Fallback by cumulative *residual* target variance
    target = float(args.target_variance)
    cum_eff = np.cumsum(var_for_knee)
    k_eff_target = int(np.searchsorted(cum_eff, target) + 1)
    k_eff_target = min(k_eff_target, len(cum_eff))
    k_target = drop_top + k_eff_target

    logging.info(
        f"Elbow k (method={args.method}, scree after drop_top={drop_top}): {k_elbow} | "
        f"var_at_elbow={var_ratio[k_elbow-1]:.6f}"
    )
    logging.info(
        f"Target {target:.0%} residual cumulative variance k: {k_target} | "
        f"cum_var_residual={cum_eff[k_eff_target-1]:.6f}"
    )

    # Save decision JSON
    decision = {
        "model": str(model_path),
        "vocab": str(vocab_path),
        "fit_on": args.fit_on,
        "fit_topn": int(args.fit_topn),
        "embedding_dim": int(emb_dim),
        "n_fit_vectors": int(n_fit),
        "max_components": int(max_comps),
        "method": args.method,
        "drop_top": int(drop_top),
        "k_elbow_total": int(k_elbow),
        "k_elbow_effective": int(k_eff_elbow),
        "var_at_elbow": float(var_ratio[k_elbow - 1]),
        "k_target_total": int(k_target),
        "k_target_effective": int(k_eff_target),
        "target_variance_residual": float(target),
        "cum_var_residual_at_target": float(cum_eff[k_eff_target - 1]),
    }
    (outdir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Save scree plot
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(1, len(var_ratio) + 1)
    plt.plot(x, var_ratio, marker='o', linewidth=1)
    if drop_top > 0:
        plt.axvspan(1, drop_top, alpha=0.1)
    plt.axvline(k_elbow, linestyle='--')
    plt.axhline(var_ratio[k_elbow - 1], linestyle='--')
    plt.xlabel("# PCA components")
    plt.ylabel("Per-component explained variance (scree)")
    ttl = "PCA scree plot (per-component variance)"
    if drop_top > 0:
        ttl += f" — drop_top={drop_top}"
    plt.title(ttl)
    plt.tight_layout()
    fig.savefig(outdir / "scree_variance.png", dpi=160)
    plt.close(fig)

    # Persist PCA transformer (full max_components basis)
    dump(pca, outdir / "pca.joblib")

    # Project your tokens using the fitted basis; then slice off top PCs
    Z_full = pca.transform(X_subset)  # (n_tokens, max_components)

    def _project_and_save(k_total: int, fname: str):
        k_eff = max(0, k_total - drop_top)
        if k_eff <= 0:
            raise SystemExit("Requested k minus drop_top <= 0; nothing to keep.")
        Z = Z_full[:, drop_top: drop_top + k_eff]
        np.savez_compressed(outdir / fname, tokens=np.array(in_vocab, dtype=object), Z=Z)

    _project_and_save(k_elbow, "reduced_embeddings_elbow.npz")
    if k_target != k_elbow:
        _project_and_save(k_target, "reduced_embeddings_target.npz")

    logging.info("Done.")


if __name__ == "__main__":
    main()
