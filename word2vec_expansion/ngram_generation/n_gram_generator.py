#!/usr/bin/env python3
"""
Phrase miner + parameter search (multiprocessing) to maximize coverage of annotated n-grams.

Key optimizations
-----------------
- Parallel mining of 1..3-gram counts/doc-freq using batched workers.
- Parallel scoring for each (lambda_len, lambda_idf, lambda_diff) grid point.
- Parallel writing of the best materialized corpus.
- Weight-cache and candidate sets are precomputed and shipped to workers.

Usage
-----
python n_gram_generator.py \
  --target_json path/to/seed_terms_minX.json \
  --outdir out/miner_run \
  --min_count 5 \
  --generate_lengths 2          # or 3, or 2,3
  --grid_lambda_len 0.0,0.1,1.0 \
  --grid_lambda_idf 1.0 \
  --grid_lambda_diff 0.0,0.5,1.0 \
  --workers 0                   # 0 => use os.cpu_count()-1
  --batch_size 1000             # docs per task
  --write_best_corpus

Notes
-----
- Unigrams are *always* selectable as fallback; 2/3-gram generation is controlled by --generate_lengths.
- No sentence boundary constraints. Docs come pre-tokenized via process_all_noburp().
- Cohesion gate is stubbed (always True) so you can flip it on later.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

# --- project imports ---
sys.path.append("../vocal_disorder")
from utils.text_pipeline import process_text
from utils.load_json import load_json
from utils.load_and_process_docs import process_all_noburp


# ========================== Utilities ==========================

def compute_idf(N_docs: int, df: int) -> float:
    """Smooth IDF: log((N+1)/(df+1)) + 1."""
    return math.log((N_docs + 1.0) / (df + 1.0)) + 1.0


@dataclass
class Interval:
    start: int
    end: int
    key: str
    n: int
    w: float


def weighted_interval_schedule(intervals: List[Interval]) -> List[Interval]:
    """Classic weighted interval scheduling for optimal non-overlapping selection."""
    if not intervals:
        return []
    ivals = sorted(intervals, key=lambda x: x.end)
    ends = [iv.end for iv in ivals]
    import bisect
    p = []
    for j, iv in enumerate(ivals):
        i = bisect.bisect_right(ends, iv.start) - 1
        p.append(i)
    n = len(ivals)
    M = [0.0] * (n + 1)
    for j in range(1, n + 1):
        incl = ivals[j-1].w + (M[p[j-1] + 1] if p[j-1] >= 0 else 0.0)
        excl = M[j-1]
        M[j] = incl if incl >= excl else excl
    sel: List[Interval] = []
    j = n
    while j > 0:
        incl = ivals[j-1].w + (M[p[j-1] + 1] if p[j-1] >= 0 else 0.0)
        if incl >= M[j-1] - 1e-12:
            sel.append(ivals[j-1])
            j = p[j-1] + 1
        else:
            j -= 1
    sel.reverse()
    return sel


def normalize_key_from_tokens(toks: Sequence[str]) -> str:
    """Join tokens with a single space to make a normalized string key."""
    return " ".join(toks)


def find_all_matches(tokens: Sequence[str], grams_by_len: Dict[int, set]) -> List[Tuple[int, int, str, int]]:
    """
    Return all matches (start, end_exclusive, 'a b ...', n), allowing overlaps.
    grams_by_len[n] must contain normalized string keys like 'a b'.
    """
    matches: List[Tuple[int, int, str, int]] = []
    L = len(tokens)
    lengths = sorted(grams_by_len.keys())  # e.g., [1,2] or [1,3] or [1,2,3]
    for i in range(L):
        for n in lengths:
            if i + n <= L:
                key = " ".join(tokens[i:i+n])  # <<< string key
                if key in grams_by_len[n]:
                    matches.append((i, i+n, key, n))
    return matches


# ======================= Batching Helpers ======================

def iter_doc_batches(batch_size: int):
    """Yield lists of tokenized docs of size up to batch_size."""
    batch = []
    for toks in process_all_noburp():
        batch.append(toks)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# =================== Mining (Parallel) =========================

def _count_ngrams_batch(batch: List[List[str]]):
    counts_by_n = {1: Counter(), 2: Counter(), 3: Counter()}
    df_counts   = {1: Counter(), 2: Counter(), 3: Counter()}  # <<< count docs per term

    for toks in batch:
        L = len(toks)
        seen_doc_local = {1: set(), 2: set(), 3: set()}
        for i in range(L):
            k1 = " ".join(toks[i:i+1]); counts_by_n[1][k1] += 1; seen_doc_local[1].add(k1)
            if i + 2 <= L:
                k2 = " ".join(toks[i:i+2]); counts_by_n[2][k2] += 1; seen_doc_local[2].add(k2)
            if i + 3 <= L:
                k3 = " ".join(toks[i:i+3]); counts_by_n[3][k3] += 1; seen_doc_local[3].add(k3)
        # bump DF by +1 for each term present in THIS doc
        for n in (1, 2, 3):
            df_counts[n].update(seen_doc_local[n])

    return len(batch), {n: dict(counts_by_n[n]) for n in (1, 2, 3)}, {n: dict(df_counts[n]) for n in (1, 2, 3)}


def mine_ngrams_parallel(workers: int, batch_size: int) -> Tuple[int, Dict[int, Counter], Dict[int, Counter]]:
    """
    Parallel pass to count n-gram occurrences and doc-freqs for n=1..3.
    Returns:
      N_docs, counts_by_n, df_by_n
    """
    counts_by_n: Dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}
    df_by_n: Dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}
    N_docs = 0

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    num_workers = (os.cpu_count() - 1 if workers == 0 else workers)
    max_pending = max(2, num_workers * 2)

    with ctx.Pool(processes=num_workers) as pool:
        pending = []
        # submit & keep a small pipeline of pending tasks to avoid huge memory
        for b in iter_doc_batches(batch_size):
            pending.append(pool.apply_async(_count_ngrams_batch, (b,)))
            # throttle pending tasks
            if len(pending) >= max_pending:
                r = pending.pop(0)
                n_batch, cnts, df_cnts = r.get()
                N_docs += n_batch
                for n in (1, 2, 3):
                    counts_by_n[n].update(cnts[n])
                    df_by_n[n].update(df_cnts[n])

        # drain remaining tasks
        for r in pending:
            n_batch, cnts, df_cnts = r.get()
            N_docs += n_batch
            for n in (1, 2, 3):
                counts_by_n[n].update(cnts[n])
                df_by_n[n].update(df_cnts[n])

    return N_docs, counts_by_n, df_by_n


# =================== Model Build ===============================

def build_phrase_model(N_docs: int,
                       counts_by_n: Dict[int, Counter],
                       df_by_n: Dict[int, Counter],
                       min_count: int) -> Dict:
    """
    Build a model object with IDFs and candidate sets.
    - Keeps 2/3-gram candidates with count >= min_count.
    - Unigrams are kept for IDF computation and selection fallback (not gated by min_count).
    """
    idf_1 = {k: compute_idf(N_docs, df_by_n[1][k]) for k in df_by_n[1].keys()}
    cand_2 = {k for k, c in counts_by_n[2].items() if c >= min_count}
    cand_3 = {k for k, c in counts_by_n[3].items() if c >= min_count}
    idf_2 = {k: compute_idf(N_docs, df_by_n[2][k]) for k in cand_2}
    idf_3 = {k: compute_idf(N_docs, df_by_n[3][k]) for k in cand_3}

    model = {
        "meta": {
            "N_docs": N_docs,
            "min_count": min_count,
            "idf_smoothing": "log((N+1)/(df+1))+1",
            "note": "Unigrams not gated by min_count; phrases (2/3) are."
        },
        "counts": {
            "1": counts_by_n[1],
            "2": {k: counts_by_n[2][k] for k in cand_2},
            "3": {k: counts_by_n[3][k] for k in cand_3},
        },
        "df": {
            "1": df_by_n[1],
            "2": {k: df_by_n[2][k] for k in cand_2},
            "3": {k: df_by_n[3][k] for k in cand_3},
        },
        "idf": {
            "1": idf_1,
            "2": idf_2,
            "3": idf_3,
        },
        "candidates_by_len": {
            "1": set(df_by_n[1].keys()),  # for selection fallback
            "2": cand_2,
            "3": cand_3,
        }
    }
    return model


# =================== Target (Annotated) Loading =================

def load_target_counts(path: str) -> Dict[str, int]:
    """
    Load target counts JSON and normalize keys with process_text.
    Expects an object with "seed_term_counts" mapping term -> count.
    Returns mapping from normalized "a b ..." -> int.
    """
    data = load_json(path)
    raw = data.get("seed_term_counts", {})
    out: Dict[str, int] = {}
    for term, cnt in raw.items():
        toks = process_text(term)  # consistent normalization with your pipeline
        if 1 <= len(toks) <= 3:
            key = normalize_key_from_tokens(toks)
            out[key] = int(cnt)
    return out


# =================== Weighting & Selection =====================

def cohesion_gate_pass(term: str) -> bool:
    """Placeholder for optional cohesion gating (PMI/LLR). Currently always True."""
    return True


def weight_of_cached(term_key: str, n: int, idf1: Dict[str, float], idf2: Dict[str, float], idf3: Dict[str, float],
                     lam_len: float, lam_idf: float, lam_diff: float) -> float:
    """Fast weight using raw dicts; avoids building model slices in workers."""
    if n == 1:
        idf_ng = idf1.get(term_key, 0.0)
    elif n == 2:
        idf_ng = idf2.get(term_key, 0.0)
    else:
        idf_ng = idf3.get(term_key, 0.0)
    parts = term_key.split()
    sum_idf_uni = sum(idf1.get(p, 0.0) for p in parts)
    return lam_len * float(n) + lam_idf * idf_ng + lam_diff * (idf_ng - sum_idf_uni)


def precompute_weight_cache(model: Dict,
                            lambdas: Tuple[float, float, float],
                            generate_lengths: set[int]) -> Dict[str, float]:
    """Precompute weight(term) for all unigrams + allowed 2/3-grams."""
    lam_len, lam_idf, lam_diff = lambdas
    idf1, idf2, idf3 = model["idf"]["1"], model["idf"]["2"], model["idf"]["3"]
    weights: Dict[str, float] = {}

    # unigrams
    for key in model["candidates_by_len"]["1"]:
        idf_ng = idf1.get(key, 0.0)
        weights[key] = lam_len * 1.0 + lam_idf * idf_ng + lam_diff * (idf_ng - idf_ng)  # diff==0 for unigram

    if 2 in generate_lengths:
        for key in model["candidates_by_len"]["2"]:
            parts = key.split()
            idf_ng = idf2.get(key, 0.0)
            sum_idf_uni = sum(idf1.get(p, 0.0) for p in parts)
            weights[key] = lam_len * 2.0 + lam_idf * idf_ng + lam_diff * (idf_ng - sum_idf_uni)

    if 3 in generate_lengths:
        for key in model["candidates_by_len"]["3"]:
            parts = key.split()
            idf_ng = idf3.get(key, 0.0)
            sum_idf_uni = sum(idf1.get(p, 0.0) for p in parts)
            weights[key] = lam_len * 3.0 + lam_idf * idf_ng + lam_diff * (idf_ng - sum_idf_uni)

    return weights


def build_grams_by_len_for_selection(model: Dict, generate_lengths: set[int]) -> Dict[int, set]:
    """
    Build selection set:
      - Unigrams: ALL observed unigrams (fallback, always included)
      - 2/3-grams: only candidates that met min_count and pass cohesion gate,
                   filtered by --generate_lengths (e.g., {2} or {2,3}).
    """
    grams_by_len = {1: set(model["candidates_by_len"]["1"])}
    if 2 in generate_lengths:
        grams_by_len[2] = {k for k in model["candidates_by_len"]["2"] if cohesion_gate_pass(k)}
    if 3 in generate_lengths:
        grams_by_len[3] = {k for k in model["candidates_by_len"]["3"] if cohesion_gate_pass(k)}
    return grams_by_len


# =================== Parallel Scoring Materialization ==========

def _score_batch(args):
    """
    Worker: score a batch for given lambdas & generate_lengths.
    Returns: (achieved_per_term dict, achieved_sum int, upper_bound_sum int)
    """
    batch, grams_by_len, weights, target_set = args
    achieved = Counter()
    achieved_sum = 0
    upper_sum = 0

    for toks in batch:
        matches = find_all_matches(toks, grams_by_len)
        intervals = [Interval(s, e, key, n, weights.get(key, 0.0)) for (s, e, key, n) in matches]
        chosen = weighted_interval_schedule(intervals)
        for iv in chosen:
            if iv.key in target_set:
                achieved[iv.key] += 1
                achieved_sum += 1
        # upper bound (annotated only, weight=1)
        ann_intervals = [Interval(s, e, key, n, 1.0) for (s, e, key, n) in matches if key in target_set]
        upper_sum += len(weighted_interval_schedule(ann_intervals))

    return dict(achieved), achieved_sum, upper_sum


def score_run_parallel(model: Dict,
                       lambdas: Tuple[float, float, float],
                       target_counts: Dict[str, int],
                       generate_lengths: set[int],
                       workers: int,
                       batch_size: int) -> Tuple[int, Dict[str, int], int, int]:
    """
    Parallel scoring pass for one lambda triple.
    """
    grams_by_len = build_grams_by_len_for_selection(model, generate_lengths)
    weights = precompute_weight_cache(model, lambdas, generate_lengths)
    target_set = set(target_counts.keys())

    total_docs = 0
    achieved_per_term: Dict[str, int] = Counter()
    achieved_sum = 0
    upper_bound_sum = 0

    pool = mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(
        processes=(os.cpu_count() - 1 if workers == 0 else workers)
    )
    try:
        async_results = []
        for b in iter_doc_batches(batch_size):
            total_docs += len(b)
            async_results.append(pool.apply_async(_score_batch, ((b, grams_by_len, weights, target_set),)))
        for r in async_results:
            d, a, u = r.get()
            achieved_per_term.update(d)
            achieved_sum += a
            upper_bound_sum += u
    finally:
        pool.close()
        pool.join()

    return total_docs, achieved_per_term, achieved_sum, upper_bound_sum


def _materialize_batch(args):
    """
    Worker: materialize a batch to tokens with underscores.
    Returns list of json lines for writing and (achieved_per_term dict) is NOT needed here.
    """
    base_index, batch, grams_by_len, weights = args
    lines = []
    for i, toks in enumerate(batch):
        matches = find_all_matches(toks, grams_by_len)
        intervals = [Interval(s, e, key, n, weights.get(key, 0.0)) for (s, e, key, n) in matches]
        chosen = weighted_interval_schedule(intervals)

        out = []
        start2iv = {iv.start: iv for iv in chosen}
        j = 0
        while j < len(toks):
            if j in start2iv:
                iv = start2iv[j]
                out.append("_".join(iv.key.split()) if iv.n > 1 else iv.key)
                j = iv.end
            else:
                out.append(toks[j])
                j += 1
        lines.append(json.dumps({"doc_index": base_index + i, "tokens": out}, ensure_ascii=False))
    return lines


def write_materialized_corpus_parallel(model: Dict,
                                       lambdas: Tuple[float, float, float],
                                       generate_lengths: set[int],
                                       out_path: Path,
                                       workers: int,
                                       batch_size: int):
    """
    Parallel materialization over full corpus, writing JSONL.
    """
    grams_by_len = build_grams_by_len_for_selection(model, generate_lengths)
    weights = precompute_weight_cache(model, lambdas, generate_lengths)

    pool = mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(
        processes=(os.cpu_count() - 1 if workers == 0 else workers)
    )

    base_index = 0
    try:
        with out_path.open("w", encoding="utf-8") as f_out:
            async_results = []
            for b in iter_doc_batches(batch_size):
                async_results.append(pool.apply_async(_materialize_batch, ((base_index, b, grams_by_len, weights),)))
                base_index += len(b)
            # write results as they complete (unordered is fine)
            for r in async_results:
                for line in r.get():
                    f_out.write(line + "\n")
    finally:
        pool.close()
        pool.join()


# =========================== Main ==============================

def main():
    ap = argparse.ArgumentParser(description="Mine phrases and search lambdas (parallel) to maximize coverage of annotated n-grams.")
    ap.add_argument("--target_json", required=True, help="JSON with 'seed_term_counts' to cover; normalized via process_text.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_count", type=int, default=5, help="Minimum global count for 2/3-gram candidates.")
    ap.add_argument("--generate_lengths", default="2", help="Comma-separated n for phrase generation (e.g., '2' or '2,3'). Unigrams are always included as fallback.")
    ap.add_argument("--grid_lambda_len", default="0.0,0.1,1.0", help="Comma-separated values.")
    ap.add_argument("--grid_lambda_idf", default="1.0", help="Comma-separated values.")
    ap.add_argument("--grid_lambda_diff", default="0.0,0.5,1.0", help="Comma-separated values.")
    ap.add_argument("--workers", type=int, default=0, help="Number of processes (0 => os.cpu_count()-1).")
    ap.add_argument("--batch_size", type=int, default=2000, help="Docs per task for multiprocessing.")
    ap.add_argument("--write_best_corpus", action="store_true", help="Write the best materialized corpus JSONL in parallel.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    # Parse allowed generation lengths
    generate_lengths = {int(x) for x in args.generate_lengths.split(",") if x.strip()}
    for n in list(generate_lengths):
        if n not in (2, 3):
            raise SystemExit(f"--generate_lengths may contain only 2 and/or 3 (got {n}).")

    # --------- Mine counts + model on full corpus (parallel) ---------
    print(f"Mining 1..3-grams (full corpus) with {('auto' if args.workers==0 else args.workers)} workers...", flush=True)
    N_docs, counts_by_n, df_by_n = mine_ngrams_parallel(args.workers, args.batch_size)
    model = build_phrase_model(N_docs, counts_by_n, df_by_n, args.min_count)

    # Save phrase model for reuse
    model_path = outdir / f"{ts}_phrase_model.json"
    model_json = {
        "meta": model["meta"],
        "counts": {n: dict(model["counts"][n]) for n in ("1", "2", "3")},
        "df": {n: dict(model["df"][n]) for n in ("1", "2", "3")},
        "idf": {n: dict(model["idf"][n]) for n in ("1", "2", "3")},
        "candidates_by_len": {
            "1": list(model["candidates_by_len"]["1"]),
            "2": list(model["candidates_by_len"]["2"]),
            "3": list(model["candidates_by_len"]["3"]),
        }
    }
    model_json["meta"]["generate_lengths_default"] = sorted(generate_lengths)
    model_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")
    print(f"Saved phrase model -> {model_path}")

    # --------- Load target counts (normalized) ---------
    target_counts = load_target_counts(args.target_json)
    target_total = int(sum(target_counts.values()))
    print(f"Loaded target terms: {len(target_counts)} | target total count: {target_total}")

    # --------- Grid search over lambdas (parallel per setting) ---------
    grid_len = [float(x) for x in args.grid_lambda_len.split(",") if x.strip() != ""]
    grid_idf = [float(x) for x in args.grid_lambda_idf.split(",") if x.strip() != ""]
    grid_diff = [float(x) for x in args.grid_lambda_diff.split(",") if x.strip() != ""]
    grid_results = []
    best = None  # (achieved_sum, upper_bound_sum, lambdas, per_term_achieved)

    print(f"Running grid search (generate_lengths={sorted(generate_lengths)})...")
    for lam_len in grid_len:
        for lam_idf in grid_idf:
            for lam_diff in grid_diff:
                lambdas = (lam_len, lam_idf, lam_diff)
                total_docs, per_term_achieved, achieved_sum, upper_bound_sum = score_run_parallel(
                    model, lambdas, target_counts, generate_lengths, args.workers, args.batch_size
                )
                coverage_ratio = achieved_sum / target_total if target_total > 0 else 1.0
                ub_ratio = achieved_sum / upper_bound_sum if upper_bound_sum > 0 else 1.0

                grid_results.append({
                    "lambdas": {"lambda_len": lam_len, "lambda_idf": lam_idf, "lambda_diff": lam_diff},
                    "generate_lengths": sorted(generate_lengths),
                    "docs": total_docs,
                    "achieved_sum": achieved_sum,
                    "upper_bound_sum": upper_bound_sum,
                    "coverage_ratio_vs_target": coverage_ratio,
                    "coverage_ratio_vs_upper_bound": ub_ratio
                })

                # tie-break by upper-bound ratio (fixing previous bug)
                prev_ub_ratio = (best[0] / best[1]) if (best and best[1] > 0) else -1.0
                if (best is None) or (achieved_sum > best[0]) or (achieved_sum == best[0] and ub_ratio > prev_ub_ratio):
                    best = (achieved_sum, upper_bound_sum, lambdas, per_term_achieved)

                print(f"  λ_len={lam_len:.3f} λ_idf={lam_idf:.3f} λ_diff={lam_diff:.3f} "
                      f"=> achieved={achieved_sum} / ub={upper_bound_sum} "
                      f"(vs_target={coverage_ratio:.4f}, vs_ub={ub_ratio:.4f})")

    grid_path = outdir / f"{ts}_grid_results.json"
    grid_path.write_text(json.dumps(grid_results, indent=2), encoding="utf-8")
    print(f"Saved grid results -> {grid_path}")

    # --------- Write coverage report for best setting ---------
    assert best is not None
    best_achieved, best_upper, best_lambdas, best_per_term = best
    report = {
        "meta": {
            "timestamp": ts,
            "N_docs": model["meta"]["N_docs"],
            "min_count_phrases": args.min_count,
            "generate_lengths": sorted(generate_lengths),
            "lambdas": {
                "lambda_len": best_lambdas[0],
                "lambda_idf": best_lambdas[1],
                "lambda_diff": best_lambdas[2],
            },
            "idf_smoothing": model["meta"]["idf_smoothing"],
            "notes": "Upper bound computed via WIS restricted to annotated terms with weight=1."
        },
        "target": {
            "num_terms": len(target_counts),
            "total_count": target_total
        },
        "scores": {
            "achieved_sum": int(best_achieved),
            "upper_bound_sum": int(best_upper),
            "coverage_ratio_vs_target": float(best_achieved / target_total) if target_total > 0 else 1.0,
            "coverage_ratio_vs_upper_bound": float(best_achieved / best_upper) if best_upper > 0 else 1.0
        },
        "per_term": {}
    }
    for k, v in sorted(best_per_term.items(), key=lambda kv: (-kv[1], kv[0])):
        report["per_term"][k] = {"achieved": int(v), "target": int(target_counts.get(k, 0))}
    report_path = outdir / f"{ts}_coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved coverage report -> {report_path}")

    # --------- Optionally write the best materialized corpus (parallel) ---------
    if args.write_best_corpus:
        phrased_path = outdir / f"{ts}_best_phrased_corpus.jsonl"
        print("Materializing best corpus in parallel...")
        write_materialized_corpus_parallel(model, best_lambdas, generate_lengths, phrased_path, args.workers, args.batch_size)
        print(f"Saved best materialized corpus -> {phrased_path}")

    # Save the model (include default lengths for convenience)
    model_json["meta"]["generate_lengths_default"] = sorted(generate_lengths)
    model_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")

    print("Done.")


# ====================== Import helpers ==========================

def load_phrase_model(path: str) -> Dict:
    """Load a phrase model JSON saved by this script (for reuse elsewhere)."""
    data = load_json(path)
    # Rebuild sets for candidates
    data["candidates_by_len"]["1"] = set(data["candidates_by_len"]["1"])
    data["candidates_by_len"]["2"] = set(data["candidates_by_len"]["2"])
    data["candidates_by_len"]["3"] = set(data["candidates_by_len"]["3"])
    return data


if __name__ == "__main__":
    main()
