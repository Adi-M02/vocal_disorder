"""
Phrase miner + parameter search (multiprocessing) with PMI/LLR-enhanced selection
to maximize coverage of annotated 1–2 grams.

What’s new
----------
- Global PMI (log2) and LLR (Dunning G^2) are computed for bigrams.
- Selection weight adds: λ_pmi * PMI_log2 + λ_llr * log2(1 + LLR_G2).
- Grid search sweeps λ_pmi and λ_llr (in addition to your existing λ’s).
- Everything else (parallel mining, WIS selection, coverage scoring/UB, JSON outputs) preserved.

Usage
-----
python n_gram_generator_pmi_llr.py \
  --target_json path/to/seed_term_counts.json \
  --outdir out/miner_run \
  --min_count 5 \
  --generate_lengths 2 \
  --grid_lambda_len 0.0,0.1,1.0 \
  --grid_lambda_idf 1.0 \
  --grid_lambda_diff -1.0,0.0,0.5 \
  --grid_lambda_boundary 0,0.25,0.5 \
  --grid_lambda_pmi 0,0.5,1.0,2.0 \
  --grid_lambda_llr 0,0.05,0.1 \
  --workers 0 \
  --batch_size 2000 \
  --write_best_corpus

Notes
-----
- PMI/LLR applied to bigrams only. For trigrams, these terms are 0 (future extension possible).
- Unigrams always selectable; 2/3-grams gated by --min_count (you can still force-include annotated if desired).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import multiprocessing as mp
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

# --- project imports ---
sys.path.append("../vocal_disorder")
from utils.text_pipeline import process_text
from utils.load_json import load_json
from utils.load_and_process_docs import process_all_noburp


# ========================== Utilities ==========================

def log2(x: float) -> float:
    return math.log(x, 2.0)

def log2_1p(x: float) -> float:
    return math.log1p(x) / math.log(2.0)

def compute_idf_log2(N_docs: int, df: int) -> float | None:
    """IDF with base-2: log2(N / df). None if df<=0."""
    if df <= 0:
        return None
    return log2(N_docs / float(df))

def compute_idf_smooth(N_docs: int, df: int) -> float:
    """Original smooth IDF kept for metadata."""
    return math.log((N_docs + 1.0) / (df + 1.0)) + 1.0


@dataclass
class Interval:
    start: int
    end: int
    key: str
    n: int
    w: float


def weighted_interval_schedule(intervals: List[Interval]) -> List[Interval]:
    """Optimal non-overlapping selection."""
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
    return " ".join(toks)


def find_all_matches(tokens: Sequence[str], grams_by_len: Dict[int, set]) -> List[Tuple[int, int, str, int]]:
    """Return matches (start, end_excl, key, n)."""
    matches: List[Tuple[int, int, str, int]] = []
    L = len(tokens)
    lengths = sorted(grams_by_len.keys())
    for i in range(L):
        for n in lengths:
            if i + n <= L:
                key = " ".join(tokens[i:i+n])
                if key in grams_by_len[n]:
                    matches.append((i, i+n, key, n))
    return matches


# ======================= Batching Helpers ======================

def iter_doc_batches(batch_size: int):
    """Yield lists of tokenized docs up to batch_size."""
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
    """
    Worker: count 1–3-gram TF, DF; plus bigram-first/second and N_bigrams for PMI/LLR.
    Returns:
      n_docs, counts_by_n dicts, df_by_n dicts, first_counts, second_counts, N_bigrams
    """
    counts_by_n = {1: Counter(), 2: Counter(), 3: Counter()}
    df_counts   = {1: Counter(), 2: Counter(), 3: Counter()}
    first_counts = Counter()   # token as first in any bigram
    second_counts = Counter()  # token as second in any bigram
    total_bigrams = 0

    for toks in batch:
        L = len(toks)
        seen_doc_local = {1: set(), 2: set(), 3: set()}

        for i in range(L):
            k1 = " ".join(toks[i:i+1])
            counts_by_n[1][k1] += 1
            seen_doc_local[1].add(k1)
            if i + 2 <= L:
                k2 = " ".join(toks[i:i+2])
                counts_by_n[2][k2] += 1
                seen_doc_local[2].add(k2)
            if i + 3 <= L:
                k3 = " ".join(toks[i:i+3])
                counts_by_n[3][k3] += 1
                seen_doc_local[3].add(k3)

        # bigram adjacency totals for PMI/LLR
        if L >= 2:
            total_bigrams += (L - 1)
            for i in range(L - 1):
                a, b = toks[i], toks[i+1]
                first_counts[a] += 1
                second_counts[b] += 1

        for n in (1, 2, 3):
            df_counts[n].update(seen_doc_local[n])

    return (
        len(batch),
        {n: dict(counts_by_n[n]) for n in (1, 2, 3)},
        {n: dict(df_counts[n]) for n in (1, 2, 3)},
        dict(first_counts),
        dict(second_counts),
        int(total_bigrams),
    )


def mine_ngrams_parallel(workers: int, batch_size: int):
    """Parallel pass; returns N_docs, counts_by_n, df_by_n, first_counts, second_counts, N_bigrams."""
    counts_by_n = {1: Counter(), 2: Counter(), 3: Counter()}
    df_by_n     = {1: Counter(), 2: Counter(), 3: Counter()}
    first_counts = Counter()
    second_counts = Counter()
    N_bigrams = 0
    N_docs = 0

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    num_workers = (os.cpu_count() - 1 if workers == 0 else workers)
    max_pending = max(2, num_workers * 2)

    with ctx.Pool(processes=num_workers) as pool:
        pending = []
        for b in iter_doc_batches(batch_size):
            pending.append(pool.apply_async(_count_ngrams_batch, (b,)))
            if len(pending) >= max_pending:
                n_batch, cnts, dfs, fcnt, scnt, nbi = pending.pop(0).get()
                N_docs += n_batch
                for n in (1, 2, 3):
                    counts_by_n[n].update(cnts[n])
                    df_by_n[n].update(dfs[n])
                first_counts.update(fcnt)
                second_counts.update(scnt)
                N_bigrams += nbi
        for r in pending:
            n_batch, cnts, dfs, fcnt, scnt, nbi = r.get()
            N_docs += n_batch
            for n in (1, 2, 3):
                counts_by_n[n].update(cnts[n])
                df_by_n[n].update(dfs[n])
            first_counts.update(fcnt)
            second_counts.update(scnt)
            N_bigrams += nbi

    return N_docs, counts_by_n, df_by_n, first_counts, second_counts, N_bigrams


# =================== PMI / LLR for bigrams =====================

def compute_pmi_llr_for_bigrams(counts_bi: Counter, first_counts: Counter, second_counts: Counter, N_bigrams: int):
    """
    Return two dicts: pmi_log2[k], llr_g2[k] for each bigram string key k="a b".
    """
    pmi = {}
    llr = {}

    def xlogx(x: float) -> float:
        return 0.0 if x <= 0 else x * math.log(x)

    for k, k11 in counts_bi.items():
        try:
            a, b = k.split()
        except ValueError:
            continue
        first_a = int(first_counts.get(a, 0))
        second_b = int(second_counts.get(b, 0))

        # PMI log2
        if k11 > 0 and first_a > 0 and second_b > 0 and N_bigrams > 0:
            pmi[k] = log2((k11 * float(N_bigrams)) / (float(first_a) * float(second_b)))
        else:
            pmi[k] = 0.0

        # LLR G^2
        k12 = max(0, first_a - k11)
        k21 = max(0, second_b - k11)
        k22 = max(0, N_bigrams - k11 - k12 - k21)
        N = k11 + k12 + k21 + k22
        if N <= 0:
            llr[k] = 0.0
        else:
            row1 = k11 + k12
            row2 = k21 + k22
            col1 = k11 + k21
            col2 = k12 + k22
            m11 = row1 * col1 / N if N else 0.0
            m12 = row1 * col2 / N if N else 0.0
            m21 = row2 * col1 / N if N else 0.0
            m22 = row2 * col2 / N if N else 0.0
            if min(m11, m12, m21, m22) <= 0:
                llr[k] = 0.0
            else:
                observed = xlogx(k11) + xlogx(k12) + xlogx(k21) + xlogx(k22)
                expected = xlogx(m11) + xlogx(m12) + xlogx(m21) + xlogx(m22)
                g2 = 2.0 * (observed - expected)
                llr[k] = float(max(0.0, g2))

    return pmi, llr


# =================== Model Build ===============================

def build_phrase_model(N_docs: int,
                       counts_by_n: Dict[int, Counter],
                       df_by_n: Dict[int, Counter],
                       min_count: int,
                       pmi_log2_bi: Dict[str, float],
                       llr_g2_bi: Dict[str, float]) -> Dict:
    """
    Build model with IDFs and candidate sets; attach PMI/LLR for bigram candidates.
    """
    # IDF (smooth for metadata + base-2 per-term for selection consistency)
    idf1_smooth = {k: compute_idf_smooth(N_docs, df_by_n[1][k]) for k in df_by_n[1].keys()}
    idf2_smooth = {k: compute_idf_smooth(N_docs, df_by_n[2][k]) for k in df_by_n[2].keys()}
    idf3_smooth = {k: compute_idf_smooth(N_docs, df_by_n[3][k]) for k in df_by_n[3].keys()}

    idf1_log2 = {k: compute_idf_log2(N_docs, df_by_n[1][k]) or 0.0 for k in df_by_n[1].keys()}
    idf2_log2_all = {k: compute_idf_log2(N_docs, df_by_n[2][k]) or 0.0 for k in df_by_n[2].keys()}
    idf3_log2_all = {k: compute_idf_log2(N_docs, df_by_n[3][k]) or 0.0 for k in df_by_n[3].keys()}

    # candidates (2/3) by min_count
    cand_2 = {k for k, c in counts_by_n[2].items() if c >= min_count}
    cand_3 = {k for k, c in counts_by_n[3].items() if c >= min_count}

    # attach PMI/LLR for bigram candidates only
    pmi_bi_cand = {k: pmi_log2_bi.get(k, 0.0) for k in cand_2}
    llr_bi_cand = {k: llr_g2_bi.get(k, 0.0) for k in cand_2}

    model = {
        "meta": {
            "N_docs": N_docs,
            "min_count": min_count,
            "idf_smoothing": "log((N+1)/(df+1))+1",
            "idf_selection_base": "log2(N/df)",
            "note": "Unigrams not gated by min_count; PMI/LLR only for bigrams.",
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
        "idf_log2": {
            "1": idf1_log2,
            "2": {k: idf2_log2_all[k] for k in cand_2},
            "3": {k: idf3_log2_all[k] for k in cand_3},
        },
        "idf_smooth": {
            "1": idf1_smooth, "2": idf2_smooth, "3": idf3_smooth
        },
        "candidates_by_len": {
            "1": set(df_by_n[1].keys()),
            "2": cand_2,
            "3": cand_3,
        },
        "pmi_log2": pmi_bi_cand,     # bigrams only
        "llr_g2": llr_bi_cand,       # bigrams only
    }
    return model


# =================== Target Loading ============================

def load_target_counts(path: str) -> Dict[str, int]:
    data = load_json(path)
    raw = data.get("seed_term_counts", {})
    out: Dict[str, int] = {}
    for term, cnt in raw.items():
        toks = process_text(term)
        if 1 <= len(toks) <= 3:
            key = normalize_key_from_tokens(toks)
            out[key] = int(cnt)
    return out


# =================== Weighting & Selection =====================

def cohesion_gate_pass(term: str) -> bool:
    return True


def precompute_weight_cache(model: Dict,
                            lambdas: Tuple[float, float, float, float, float, float],
                            generate_lengths: Set[int]) -> Dict[str, float]:
    """
    Precompute weight(term) for all unigrams + allowed phrases.
    Weight uses base-2 IDF + PMI/LLR for bigrams:

      w = λ_len*n
        + λ_idf*IDF_log2(ngram)
        + λ_diff*(IDF_log2(ngram) - sum IDF_log2(unigrams))
        + λ_boundary*(n-1)
        + λ_pmi*PMI_log2(ngram)                     [bigrams only]
        + λ_llr*log2(1 + LLR_G2(ngram))             [bigrams only]
    """
    lam_len, lam_idf, lam_diff, lam_boundary, lam_pmi, lam_llr = lambdas
    idf1 = model["idf_log2"]["1"]
    idf2 = model["idf_log2"]["2"]
    idf3 = model["idf_log2"]["3"]
    pmi2 = model["pmi_log2"]
    llr2 = model["llr_g2"]

    weights: Dict[str, float] = {}

    # unigrams
    for key in model["candidates_by_len"]["1"]:
        idf_ng = idf1.get(key, 0.0)
        w = lam_len * 1.0 + lam_idf * idf_ng  # diff term = 0 for unigram
        weights[key] = w

    # bigrams
    if 2 in generate_lengths:
        for key in model["candidates_by_len"]["2"]:
            idf_ng = idf2.get(key, 0.0)
            parts = key.split()
            sum_idf_uni = sum(idf1.get(p, 0.0) for p in parts)
            w = (lam_len * 2.0
                 + lam_idf * idf_ng
                 + lam_diff * (idf_ng - sum_idf_uni)
                 + lam_boundary * 1.0
                 + lam_pmi * pmi2.get(key, 0.0)
                 + lam_llr * log2_1p(llr2.get(key, 0.0)))
            weights[key] = w

    # trigrams (PMI/LLR terms 0 for now)
    if 3 in generate_lengths:
        for key in model["candidates_by_len"]["3"]:
            idf_ng = idf3.get(key, 0.0)
            parts = key.split()
            sum_idf_uni = sum(idf1.get(p, 0.0) for p in parts)
            w = (lam_len * 3.0
                 + lam_idf * idf_ng
                 + lam_diff * (idf_ng - sum_idf_uni)
                 + lam_boundary * 2.0)
            weights[key] = w

    return weights


def build_grams_by_len_for_selection(model: Dict, generate_lengths: Set[int]) -> Dict[int, set]:
    grams_by_len = {1: set(model["candidates_by_len"]["1"])}
    if 2 in generate_lengths:
        grams_by_len[2] = {k for k in model["candidates_by_len"]["2"] if cohesion_gate_pass(k)}
    if 3 in generate_lengths:
        grams_by_len[3] = {k for k in model["candidates_by_len"]["3"] if cohesion_gate_pass(k)}
    return grams_by_len


# =================== Parallel Scoring & Materialize ============

# worker globals (initializer)
_G_GRAMS_BY_LEN = None
_G_WEIGHTS = None
_G_ANN_SET = None
_G_ANN_ONLY_BY_LEN = None

def _init_worker(grams_by_len, weights, annotated_set, ann_only_by_len):
    global _G_GRAMS_BY_LEN, _G_WEIGHTS, _G_ANN_SET, _G_ANN_ONLY_BY_LEN
    _G_GRAMS_BY_LEN = grams_by_len
    _G_WEIGHTS = weights
    _G_ANN_SET = annotated_set
    _G_ANN_ONLY_BY_LEN = ann_only_by_len


def _score_batch(docs: List[List[str]]):
    achieved = Counter(); achieved_sum = 0; upper_sum = 0
    for toks in docs:
        # achieved (all candidates with weights)
        matches = find_all_matches(toks, _G_GRAMS_BY_LEN)
        ivals = [Interval(s, e, key, n, _G_WEIGHTS.get(key, 0.0)) for (s, e, key, n) in matches]
        chosen = weighted_interval_schedule(ivals)
        for iv in chosen:
            if iv.key in _G_ANN_SET:
                achieved[iv.key] += 1; achieved_sum += 1

        # upper bound: annotated-only, weight=1, not gated by candidates
        ann_matches = find_all_matches(toks, _G_ANN_ONLY_BY_LEN)
        ann_ivals = [Interval(s, e, key, n, 1.0) for (s, e, key, n) in ann_matches]
        upper_sum += len(weighted_interval_schedule(ann_ivals))
    return dict(achieved), achieved_sum, upper_sum


def score_run_parallel(model: Dict,
                       lambdas: Tuple[float, float, float, float, float, float],
                       target_counts: Dict[str, int],
                       generate_lengths: Set[int],
                       workers: int,
                       batch_size: int) -> Tuple[int, Dict[str, int], int, int]:
    grams_by_len = build_grams_by_len_for_selection(model, generate_lengths)
    weights = precompute_weight_cache(model, lambdas, generate_lengths)
    annotated_set = set(target_counts.keys())
    # Build annotated-only matcher by length (unguarded by min_count)
    ann_by_len = {1: set(), 2: set(), 3: set()}
    for k in annotated_set:
        n = len(k.split())
        if n == 1 or n in generate_lengths:
            ann_by_len[n].add(k)

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    num_workers = (os.cpu_count() - 1 if workers == 0 else workers)
    max_pending = max(2, num_workers * 2)

    total_docs = 0
    achieved_per_term = Counter(); achieved_sum = 0; upper_bound_sum = 0

    with ctx.Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(grams_by_len, weights, annotated_set, ann_by_len)) as pool:
        pending = []
        for b in iter_doc_batches(batch_size):
            total_docs += len(b)
            pending.append(pool.apply_async(_score_batch, (b,)))
            if len(pending) >= max_pending:
                d, a, u = pending.pop(0).get()
                achieved_per_term.update(d); achieved_sum += a; upper_bound_sum += u
        for r in pending:
            d, a, u = r.get()
            achieved_per_term.update(d); achieved_sum += a; upper_bound_sum += u

    return total_docs, achieved_per_term, achieved_sum, upper_bound_sum


def _materialize_batch(args):
    base_index, batch = args
    lines = []
    for i, toks in enumerate(batch):
        matches = find_all_matches(toks, _G_GRAMS_BY_LEN)
        ivals = [Interval(s, e, key, n, _G_WEIGHTS.get(key, 0.0)) for (s, e, key, n) in matches]
        chosen = weighted_interval_schedule(ivals)
        out = []
        start2iv = {iv.start: iv for iv in chosen}
        j = 0
        while j < len(toks):
            if j in start2iv:
                iv = start2iv[j]
                out.append("_".join(iv.key.split()) if iv.n > 1 else iv.key)
                j = iv.end
            else:
                out.append(toks[j]); j += 1
        lines.append(json.dumps({"doc_index": base_index + i, "tokens": out}, ensure_ascii=False))
    return lines


def write_materialized_corpus_parallel(model: Dict,
                                       lambdas: Tuple[float, float, float, float, float, float],
                                       generate_lengths: Set[int],
                                       out_path: Path,
                                       workers: int,
                                       batch_size: int):
    grams_by_len = build_grams_by_len_for_selection(model, generate_lengths)
    weights = precompute_weight_cache(model, lambdas, generate_lengths)

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    num_workers = (os.cpu_count() - 1 if workers == 0 else workers)
    max_pending = max(2, num_workers * 2)

    with ctx.Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(grams_by_len, weights, None, None)) as pool:
        base_index = 0
        pending = []
        with out_path.open("w", encoding="utf-8") as f_out:
            for b in iter_doc_batches(batch_size):
                pending.append(pool.apply_async(_materialize_batch, ((base_index, b),)))
                base_index += len(b)
                if len(pending) >= max_pending:
                    for line in pending.pop(0).get():
                        f_out.write(line + "\n")
            for r in pending:
                for line in r.get():
                    f_out.write(line + "\n")


# =========================== Main ==============================

def main():
    ap = argparse.ArgumentParser(description="PMI/LLR-enhanced phrase miner to maximize annotated 1–2-gram coverage.")
    ap.add_argument("--target_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_count", type=int, default=5, help="Minimum global count for 2/3-gram candidates.")
    ap.add_argument("--generate_lengths", default="2", help="Comma-separated (2 or 2,3). Unigrams always included.")
    # weight lambdas
    ap.add_argument("--grid_lambda_len", default="0.0", help="Comma-separated values (often cancels; keep 0).")
    ap.add_argument("--grid_lambda_idf", default="1.0", help="Comma-separated values.")
    ap.add_argument("--grid_lambda_diff", default="-1.0,0.0,0.5", help="Comma-separated values.")
    ap.add_argument("--grid_lambda_boundary", default="0,0.25,0.5", help="Comma-separated values.")
    ap.add_argument("--grid_lambda_pmi", default="0,0.5,1.0,2.0", help="Comma-separated values.")
    ap.add_argument("--grid_lambda_llr", default="0,0.05,0.1", help="Comma-separated values (applied to log2(1+LLR)).")
    # parallel
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--write_best_corpus", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    generate_lengths = {int(x) for x in args.generate_lengths.split(",") if x.strip()}
    for n in list(generate_lengths):
        if n not in (2, 3):
            raise SystemExit(f"--generate_lengths may contain only 2 and/or 3 (got {n}).")

    # --------- Mine counts + PMI/LLR (parallel) ---------
    print(f"Mining 1..3-grams + PMI/LLR (workers={('auto' if args.workers==0 else args.workers)}) ...", flush=True)
    N_docs, counts_by_n, df_by_n, first_counts, second_counts, N_bigrams = mine_ngrams_parallel(args.workers, args.batch_size)
    pmi_log2_bi, llr_g2_bi = compute_pmi_llr_for_bigrams(counts_by_n[2], first_counts, second_counts, N_bigrams)

    # --------- Build model ---------
    model = build_phrase_model(N_docs, counts_by_n, df_by_n, args.min_count, pmi_log2_bi, llr_g2_bi)

    # Save phrase model for reuse
    model_path = outdir / f"{ts}_phrase_model.json"
    model_json = {
        "meta": model["meta"] | {"N_bigrams_total": int(N_bigrams)},
        "counts": {n: dict(model["counts"][n]) for n in ("1", "2", "3")},
        "df": {n: dict(model["df"][n]) for n in ("1", "2", "3")},
        "idf_log2": {n: dict(model["idf_log2"][n]) for n in ("1", "2", "3")},
        "idf_smooth": {n: dict(model["idf_smooth"][n]) for n in ("1", "2", "3")},
        "candidates_by_len": {
            "1": list(model["candidates_by_len"]["1"]),
            "2": list(model["candidates_by_len"]["2"]),
            "3": list(model["candidates_by_len"]["3"]),
        },
        "pmi_log2": model["pmi_log2"],
        "llr_g2": model["llr_g2"],
    }
    model_json["meta"]["generate_lengths_default"] = sorted(generate_lengths)
    model_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")
    print(f"Saved phrase model -> {model_path}")

    # --------- Sanity check (your requested debug) ---------
    some_bi = next(iter(model["candidates_by_len"]["2"])) if model["candidates_by_len"]["2"] else None
    if some_bi:
        w0 = precompute_weight_cache(model, (0,1,0,0,0,0), {2}).get(some_bi, 0.0)   # pure IDF
        w_p = precompute_weight_cache(model, (0,1,0,0,1,0), {2}).get(some_bi, 0.0)  # +PMI
        w_l = precompute_weight_cache(model, (0,1,0,0,0,0.1), {2}).get(some_bi, 0.0)# +LLR
        print(f"[dbg] weight[{some_bi}] idf={w0:.3f} +pmi={w_p:.3f} +llr={w_l:.3f}")

    # --------- Load target counts (normalized) ---------
    target_counts = load_target_counts(args.target_json)
    target_total = int(sum(target_counts.values()))
    print(f"Loaded target terms: {len(target_counts)} | target total count: {target_total}")

    # --------- Grid search over lambdas (now includes PMI/LLR) ---------
    grid_len   = [float(x) for x in args.grid_lambda_len.split(",") if x.strip() != ""]
    grid_idf   = [float(x) for x in args.grid_lambda_idf.split(",") if x.strip() != ""]
    grid_diff  = [float(x) for x in args.grid_lambda_diff.split(",") if x.strip() != ""]
    grid_bound = [float(x) for x in args.grid_lambda_boundary.split(",") if x.strip() != ""]
    grid_pmi   = [float(x) for x in args.grid_lambda_pmi.split(",") if x.strip() != ""]
    grid_llr   = [float(x) for x in args.grid_lambda_llr.split(",") if x.strip() != ""]
    grid_results = []
    best = None  # (achieved_sum, upper_bound_sum, lambdas, per_term_achieved)

    print(f"Running grid search (generate_lengths={sorted(generate_lengths)})...")
    for lam_len in grid_len:
        for lam_idf in grid_idf:
            for lam_diff in grid_diff:
                for lam_b in grid_bound:
                    for lam_p in grid_pmi:
                        for lam_l in grid_llr:
                            lambdas = (lam_len, lam_idf, lam_diff, lam_b, lam_p, lam_l)
                            total_docs, per_term_achieved, achieved_sum, upper_bound_sum = score_run_parallel(
                                model, lambdas, target_counts, generate_lengths, args.workers, args.batch_size
                            )
                            coverage_ratio = achieved_sum / target_total if target_total > 0 else 1.0
                            ub_ratio = achieved_sum / upper_bound_sum if upper_bound_sum > 0 else 1.0

                            grid_results.append({
                                "lambdas": {
                                    "lambda_len": lam_len, "lambda_idf": lam_idf, "lambda_diff": lam_diff,
                                    "lambda_boundary": lam_b, "lambda_pmi": lam_p, "lambda_llr": lam_l
                                },
                                "generate_lengths": sorted(generate_lengths),
                                "docs": total_docs,
                                "achieved_sum": achieved_sum,
                                "upper_bound_sum": upper_bound_sum,
                                "coverage_ratio_vs_target": coverage_ratio,
                                "coverage_ratio_vs_upper_bound": ub_ratio
                            })

                            prev_ub_ratio = (best[0] / best[1]) if (best and best[1] > 0) else -1.0
                            if (best is None) or (achieved_sum > best[0]) or (achieved_sum == best[0] and ub_ratio > prev_ub_ratio):
                                best = (achieved_sum, upper_bound_sum, lambdas, per_term_achieved)

                            print(f"  λ_len={lam_len:.3f} λ_idf={lam_idf:.3f} λ_diff={lam_diff:.3f} "
                                  f"λ_b={lam_b:.3f} λ_pmi={lam_p:.3f} λ_llr={lam_l:.3f} "
                                  f"=> achieved={achieved_sum} / ub={upper_bound_sum} "
                                  f"(vs_target={coverage_ratio:.4f}, vs_ub={ub_ratio:.4f})")

    grid_path = outdir / f"{ts}_grid_results.json"
    grid_path.write_text(json.dumps(grid_results, indent=2), encoding="utf-8")
    print(f"Saved grid results -> {grid_path}")

    # --------- Best coverage report ---------
    assert best is not None
    best_achieved, best_upper, best_lambdas, best_per_term = best
    report = {
        "meta": {
            "timestamp": ts,
            "N_docs": model["meta"]["N_docs"],
            "N_bigrams_total": model_json["meta"]["N_bigrams_total"],
            "min_count_phrases": args.min_count,
            "generate_lengths": sorted(generate_lengths),
            "lambdas": {
                "lambda_len": best_lambdas[0],
                "lambda_idf": best_lambdas[1],
                "lambda_diff": best_lambdas[2],
                "lambda_boundary": best_lambdas[3],
                "lambda_pmi": best_lambdas[4],
                "lambda_llr": best_lambdas[5],
            },
            "idf_selection_base": model["meta"]["idf_selection_base"],
            "idf_smoothing": model["meta"]["idf_smoothing"],
            "notes": "Upper bound uses annotated terms only (not gated by min_count).",
        },
        "target": {
            "num_terms": len(target_counts),
            "total_count": int(sum(target_counts.values()))
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

    # --------- Optionally write best materialized corpus ---------
    if args.write_best_corpus:
        phrased_path = outdir / f"{ts}_best_phrased_corpus.jsonl"
        print("Materializing best corpus in parallel...")
        write_materialized_corpus_parallel(model, best_lambdas, generate_lengths, phrased_path, args.workers, args.batch_size)
        print(f"Saved best materialized corpus -> {phrased_path}")

    print("Done.")


# ====================== Import helpers ==========================

def load_phrase_model(path: str) -> Dict:
    """Load a phrase model JSON saved by this script (for reuse elsewhere)."""
    data = load_json(path)
    data["candidates_by_len"]["1"] = set(data["candidates_by_len"]["1"])
    data["candidates_by_len"]["2"] = set(data["candidates_by_len"]["2"])
    data["candidates_by_len"]["3"] = set(data["candidates_by_len"]["3"])
    return data


if __name__ == "__main__":
    main()
