#!/usr/bin/env python3
"""
Compute IDF (log2), PMI, and LLR for selected 1-grams and 2-grams over the full corpus.

Inputs
------
--input_json:
  - {"seed_terms": [...] } OR {"seed_term_counts": {...}}
  Terms are normalized via process_text(). Only 1-grams and 2-grams are used.
  Unigrams for any requested bigrams are auto-included.

Corpus
------
Tokenized docs from utils.load_and_process_docs.process_all_noburp()
(Stoplisting/cleaning already applied upstream.)

Outputs
-------
JSON with:
  meta: N_docs, totals, formulae, timestamps
  terms:
    "<unigram>": {
      n: 1, df: ..., idf_log2: ..., tf: ..., corpus_tfidf_log2: ...
    }
    "a b": {
      n: 2, df: ..., idf_log2: ..., tf: ..., corpus_tfidf_log2: ...,
      pmi_log2: ..., llr_g2: ...,
      idf_sum_unigrams_log2: ..., idf_delta_vs_unigrams_log2: ...
    }

Notes
-----
- IDF uses the exact formula you requested: idf = log2(N / df), no smoothing.
  If df == 0, idf_log2 is null.
- PMI is computed on the adjacent-bigram sample space:
    PMI(a,b) = log2( (k11 * N_bigrams) / (first(a) * second(b)) )
  where:
    k11       = count of adjacent bigram "a b"
    first(a)  = times 'a' appears as the first token of any bigram
    second(b) = times 'b' appears as the second token of any bigram
    N_bigrams = total adjacent bigram positions in the corpus
  If any component is 0, PMI is null.
- LLR is Dunning's G^2 on the 2x2 table:
    [[k11, k12],
     [k21, k22]]
  with:
    k12 = first(a) - k11
    k21 = second(b) - k11
    k22 = N_bigrams - k11 - k12 - k21
  Returned as a non-negative float (natural log), null if any term is undefined.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import multiprocessing as mp
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

# --- project imports ---
sys.path.append("../vocal_disorder")
from utils.text_pipeline import process_text
from utils.load_json import load_json
from utils.load_and_process_docs import process_all_noburp


# ------------------------ Helpers ------------------------

def log2_safe(x: float) -> float:
    return math.log(x, 2.0)

def idf_log2(N_docs: int, df: int):
    if df <= 0:
        return None
    return log2_safe(N_docs / float(df))

def pmi_log2_from_counts(k11: int, first_a: int, second_b: int, N_bigrams: int):
    # PMI(a,b) = log2( (k11 * N_bigrams) / (first(a) * second(b)) )
    if k11 <= 0 or first_a <= 0 or second_b <= 0 or N_bigrams <= 0:
        return None
    num = k11 * float(N_bigrams)
    den = float(first_a) * float(second_b)
    if den <= 0:
        return None
    return log2_safe(num / den)

def llr_g2(k11: int, k12: int, k21: int, k22: int):
    # Dunning (1993) G^2 (natural log)
    # Handle zeros gracefully.
    def xlogx(x: float) -> float:
        return 0.0 if x <= 0 else x * math.log(x)

    N = k11 + k12 + k21 + k22
    if N <= 0:
        return None

    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    # Expected under independence
    m11 = row1 * col1 / N if N else 0.0
    m12 = row1 * col2 / N if N else 0.0
    m21 = row2 * col1 / N if N else 0.0
    m22 = row2 * col2 / N if N else 0.0

    # If any expected is zero, LLR isn't defined well; return None
    if min(m11, m12, m21, m22) <= 0:
        return None

    observed = xlogx(k11) + xlogx(k12) + xlogx(k21) + xlogx(k22)
    expected = xlogx(m11) + xlogx(m12) + xlogx(m21) + xlogx(m22)
    g2 = 2.0 * (observed - expected)
    # numerical guard
    return float(max(0.0, g2))


# --------------- Parse & normalize input terms ---------------

def normalize_requested_terms(data: dict) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    Returns:
      target_unigrams: set of unigram string keys
      target_bigrams:  set of bigram tuple keys
    Ensures unigrams for requested bigrams are included.
    """
    raw_terms: List[str] = []
    if "seed_terms" in data and isinstance(data["seed_terms"], list):
        raw_terms = [str(t) for t in data["seed_terms"]]
    elif "seed_term_counts" in data and isinstance(data["seed_term_counts"], dict):
        raw_terms = [str(t) for t in data["seed_term_counts"].keys()]
    else:
        raise ValueError("Input JSON must have 'seed_terms' (list) or 'seed_term_counts' (object).")

    target_unigrams: Set[str] = set()
    target_bigrams: Set[Tuple[str, str]] = set()

    for term in raw_terms:
        toks = process_text(term)
        if len(toks) == 1:
            target_unigrams.add(toks[0])
        elif len(toks) == 2:
            target_bigrams.add((toks[0], toks[1]))
        # silently ignore n>2

    for a, b in list(target_bigrams):
        target_unigrams.add(a)
        target_unigrams.add(b)

    return target_unigrams, target_bigrams


# -------------------- Multiprocessing workers ------------------

_G_TARGET_UNI: Set[str] = set()
_G_TARGET_BI: Set[Tuple[str, str]] = set()

def _init_worker(target_unigrams: Set[str], target_bigrams: Set[Tuple[str, str]]):
    global _G_TARGET_UNI, _G_TARGET_BI
    _G_TARGET_UNI = target_unigrams
    _G_TARGET_BI = target_bigrams

def _doc_bigrams(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

def _stats_batch(docs: List[List[str]]):
    """
    Worker: accumulate stats for requested terms only.
    Returns:
      df_uni: dict[str,int]      (doc frequency of unigrams)
      df_bi:  dict["a b",int]    (doc frequency of bigrams)
      tf_uni: dict[str,int]      (total occurrences of unigrams)
      tf_bi:  dict["a b",int]    (total occurrences of bigrams)
      first_counts: dict[str,int]  (# times token appears as first in a bigram)
      second_counts: dict[str,int] (# times token appears as second in a bigram)
      total_bigrams: int           (sum over docs of max(0, len-1))
    """
    df_uni = Counter()
    df_bi = Counter()
    tf_uni = Counter()
    tf_bi = Counter()
    first_counts = Counter()
    second_counts = Counter()
    total_bigrams = 0

    target_uni = _G_TARGET_UNI
    target_bi = _G_TARGET_BI

    for toks in docs:
        L = len(toks)
        # --- unigrams
        if target_uni:
            uniq = set(toks)
            present_uni = uniq.intersection(target_uni)
            df_uni.update(present_uni)
            c = Counter(toks)
            for u in present_uni:
                tf_uni[u] += c[u]
        # --- bigrams
        if L >= 2 and target_bi:
            bigs = _doc_bigrams(toks)
            total_bigrams += (L - 1)
            # DF for bigrams
            uniq_bi = set(bigs)
            present_bi = uniq_bi.intersection(target_bi)
            df_bi.update(" ".join(p) for p in present_bi)
            # TF for bigrams + first/second counts for ALL positions (restricted to target unigrams)
            for (a, b) in bigs:
                if (a, b) in target_bi:
                    tf_bi[" ".join((a, b))] += 1
                if a in target_uni:
                    first_counts[a] += 1
                if b in target_uni:
                    second_counts[b] += 1
        elif L >= 2:
            total_bigrams += (L - 1)

    return (
        dict(df_uni), dict(df_bi), dict(tf_uni), dict(tf_bi),
        dict(first_counts), dict(second_counts), int(total_bigrams)
    )


# ------------------- Batch iterator over corpus ----------------

def iter_doc_batches(batch_size: int):
    batch = []
    for toks in process_all_noburp():
        batch.append(toks)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------ Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute IDF(log2), PMI, and LLR for selected 1/2-grams.")
    ap.add_argument("--input_json", required=True, help="JSON with 'seed_terms' (list) or 'seed_term_counts' (object).")
    ap.add_argument("--out_json", required=True, help="Output JSON path.")
    ap.add_argument("--workers", type=int, default=0, help="0 => use os.cpu_count()-1.")
    ap.add_argument("--batch_size", type=int, default=2000, help="Docs per task.")
    args = ap.parse_args()

    data = load_json(args.input_json)
    target_unigrams, target_bigrams = normalize_requested_terms(data)

    if not target_unigrams and not target_bigrams:
        raise SystemExit("No valid 1/2-gram terms found after normalization.")

    # Pre-build display keys for bigrams
    target_bigram_keys = {" ".join(p) for p in target_bigrams}

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    num_workers = (os.cpu_count() - 1 if args.workers == 0 else args.workers)
    max_pending = max(2, num_workers * 2)

    df_uni = Counter()
    df_bi = Counter()
    tf_uni = Counter()
    tf_bi = Counter()
    first_counts = Counter()
    second_counts = Counter()
    N_docs = 0
    N_bigrams_total = 0

    with ctx.Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(target_unigrams, target_bigrams)) as pool:
        pending = []
        for batch in iter_doc_batches(args.batch_size):
            N_docs += len(batch)
            pending.append(pool.apply_async(_stats_batch, (batch,)))
            if len(pending) >= max_pending:
                u_df, b_df, u_tf, b_tf, f_cnt, s_cnt, bi_total = pending.pop(0).get()
                df_uni.update(u_df); df_bi.update(b_df)
                tf_uni.update(u_tf); tf_bi.update(b_tf)
                first_counts.update(f_cnt); second_counts.update(s_cnt)
                N_bigrams_total += bi_total
        for r in pending:
            u_df, b_df, u_tf, b_tf, f_cnt, s_cnt, bi_total = r.get()
            df_uni.update(u_df); df_bi.update(b_df)
            tf_uni.update(u_tf); tf_bi.update(b_tf)
            first_counts.update(f_cnt); second_counts.update(s_cnt)
            N_bigrams_total += bi_total

    # Build output
    now = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    out = {
        "meta": {
            "N_docs": N_docs,
            "N_bigrams_total": int(N_bigrams_total),
            "idf_formula": "idf = log2(N_docs / df_docs)",
            "pmi_formula": "PMI(a,b) = log2( (k11 * N_bigrams) / (first(a) * second(b)) )",
            "llr_formula": "Dunning G^2 with natural log on 2x2 table",
            "generated_at": now,
            "notes": "Bigrams are contiguous pairs over the tokenized stream; DF is doc-level; TF is total occurrences.",
        },
        "terms": {}
    }

    # Unigrams
    for u in sorted(target_unigrams):
        df = int(df_uni.get(u, 0))
        tf = int(tf_uni.get(u, 0))
        idf = idf_log2(N_docs, df)
        rec = {
            "n": 1,
            "df_docs": df,
            "idf_log2": idf,
            "tf_total": tf,
            "corpus_tfidf_log2": (None if idf is None else float(tf) * idf)
        }
        out["terms"][u] = rec

    # Bigrams
    for k in sorted(target_bigram_keys):
        a, b = k.split()
        df = int(df_bi.get(k, 0))
        tf = int(tf_bi.get(k, 0))
        idf = idf_log2(N_docs, df)

        # Compare to sum of unigram idfs
        # (unigram records were already written above)
        urec_a = out["terms"].get(a, {})
        urec_b = out["terms"].get(b, {})
        idf_a = urec_a.get("idf_log2")
        idf_b = urec_b.get("idf_log2")
        idf_sum_uni = (None if (idf_a is None or idf_b is None) else float(idf_a) + float(idf_b))
        idf_delta = (None if (idf is None or idf_sum_uni is None) else float(idf) - float(idf_sum_uni))

        # PMI & LLR counts
        k11 = tf
        first_a = int(first_counts.get(a, 0))
        second_b = int(second_counts.get(b, 0))
        k12 = max(0, first_a - k11)
        k21 = max(0, second_b - k11)
        k22 = max(0, N_bigrams_total - k11 - k12 - k21)

        pmi = pmi_log2_from_counts(k11, first_a, second_b, N_bigrams_total)
        g2 = llr_g2(k11, k12, k21, k22)

        # Build bigram record, embedding the unigram metrics
        rec = {
            "n": 2,
            "df_docs": df,
            "idf_log2": idf,
            "tf_total": tf,
            "corpus_tfidf_log2": (None if idf is None else float(tf) * idf),
            "idf_sum_unigrams_log2": idf_sum_uni,
            "idf_delta_vs_unigrams_log2": idf_delta,
            "pmi_log2": pmi,
            "llr_g2": g2,
            "aux_counts": {
                "bigram_k11": k11,
                "first_a": first_a,
                "second_b": second_b,
                "k12": k12,
                "k21": k21,
                "k22": k22
            },
            "unigram_components": {
                a: {
                    "n": 1,
                    "df_docs": urec_a.get("df_docs"),
                    "idf_log2": urec_a.get("idf_log2"),
                    "tf_total": urec_a.get("tf_total"),
                    "corpus_tfidf_log2": urec_a.get("corpus_tfidf_log2"),
                },
                b: {
                    "n": 1,
                    "df_docs": urec_b.get("df_docs"),
                    "idf_log2": urec_b.get("idf_log2"),
                    "tf_total": urec_b.get("tf_total"),
                    "corpus_tfidf_log2": urec_b.get("corpus_tfidf_log2"),
                }
            }
        }
        out["terms"][k] = rec

    # Write
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote stats for {len(out['terms'])} terms to {out_path}")
    print(f"N_docs={N_docs} | N_bigrams_total={N_bigrams_total} | unigrams={len(target_unigrams)} | bigrams={len(target_bigram_keys)}")

if __name__ == "__main__":
    main()
