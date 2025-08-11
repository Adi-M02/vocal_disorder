#!/usr/bin/env python3
"""
Evaluate an input "expanded" unigram vocabulary against ground truth unigram vocabulary
on text from annotated user documents.

Inputs
------
- Expanded JSON: {"termA": [...], "termB": [...], ...}  (union of all lists = expanded set)
- Ground-truth JSON: {"seed_terms": [...]}
- return_documents(...) -> List[str]  (each item is a raw document string)

Processing
----------
- Each raw string doc is processed with process_text(text) -> List[str] tokens.
- Token-level counts:
    TP: token ∈ expanded AND token ∈ ground_truth
    FP: token ∈ expanded AND token ∉ ground_truth
    FN: token ∉ expanded AND token ∈ ground_truth
    TN: token ∉ expanded AND token ∉ ground_truth

Outputs
-------
- JSON with global metrics (precision/recall/F1/acc/specificity/balanced_acc/MCC, ROC AUC & PR AUC),
  and per-term counts for TP/FP/FN (occurrence counts).
"""

import sys, json, argparse, collections
from pathlib import Path
from typing import List, Set, Dict
from sklearn.metrics import roc_auc_score, average_precision_score

# Project imports
sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from utils.text_pipeline import process_text
from utils.load_json import load_json

# ---------------------------
# Helpers
# ---------------------------

def _normalize_unigram_set(terms: List[str]) -> Set[str]:
    """Normalize via process_text. Keep only unigrams; drop multi-token outputs."""
    out: Set[str] = set()
    dropped = 0
    for t in terms:
        toks = process_text(str(t))
        if len(toks) == 1:
            out.add(toks[0])
        elif len(toks) == 0:
            continue
        else:
            dropped += 1
    if dropped:
        print(f"[warn] dropped {dropped} terms that normalized to multi-token; unigram-only evaluation.")
    return out

def _flatten_expanded(obj) -> Set[str]:
    """Expanded JSON is {termA: [terms], termB: [terms], ...}; union all lists."""
    if not isinstance(obj, dict):
        raise ValueError("Expanded JSON must be a dict of lists.")
    vals: List[str] = []
    for v in obj.values():
        if isinstance(v, list):
            vals.extend(v)
        else:
            raise ValueError("Each expanded category must map to a list of terms.")
    return set(vals)

def _flatten_ground_truth(obj) -> Set[str]:
    """Ground-truth JSON: {'seed_terms': [...]}."""
    if not (isinstance(obj, dict) and "seed_terms" in obj and isinstance(obj["seed_terms"], list)):
        raise ValueError("Ground truth JSON must be an object with a 'seed_terms' list.")
    return _normalize_unigram_set(obj["seed_terms"])

def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0

def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return ((tp*tn - fp*fn) / (denom ** 0.5)) if denom else 0.0

def _try_metrics(y_true: List[int], y_score: List[float]) -> Dict[str, float | None]:
    """
    Compute sklearn ROC AUC & Average Precision when well-defined.
    Returns None for metrics that are undefined (e.g., empty input, single class).
    """
    out: Dict[str, float | None] = {"roc_auc": None, "avg_precision": None}
    if not y_true:
        return out
    classes = set(y_true)
    if classes == {0, 1}:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = None
    if 1 in classes:
        try:
            out["avg_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["avg_precision"] = None
    return out

# ---------------------------
# Core evaluation (token-level)
# ---------------------------

def evaluate_token_level(
    docs_raw: List[str],
    det_terms: Set[str],
    gt_terms: Set[str],
) -> Dict:
    TP = FP = FN = TN = 0
    tp_terms = collections.Counter()
    fp_terms = collections.Counter()
    fn_terms = collections.Counter()

    y_true_all: List[int] = []
    y_score_all: List[int] = []

    tokens_seen_total = 0
    docs_processed = 0
    docs_skipped_empty = 0

    for text in docs_raw:
        docs_processed += 1
        if not isinstance(text, str) or not text:
            docs_skipped_empty += 1
            continue

        toks = process_text(text)
        if not toks:  # skip empty-token docs
            docs_skipped_empty += 1
            continue

        tokens_seen_total += len(toks)

        for tok in toks:
            is_gt  = tok in gt_terms
            is_det = tok in det_terms

            if is_det and is_gt:
                TP += 1; tp_terms[tok] += 1
                y_true_all.append(1); y_score_all.append(1)
            elif is_det and not is_gt:
                FP += 1; fp_terms[tok] += 1
                y_true_all.append(0); y_score_all.append(1)
            elif (not is_det) and is_gt:
                FN += 1; fn_terms[tok] += 1
                y_true_all.append(1); y_score_all.append(0)
            else:
                TN += 1
                y_true_all.append(0); y_score_all.append(0)

    precision = _safe_div(TP, TP + FP)
    recall    = _safe_div(TP, TP + FN)
    f1        = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy  = _safe_div(TP + TN, TP + FP + FN + TN)
    specificity = _safe_div(TN, TN + FP)
    balanced_accuracy = 0.5 * (recall + specificity)
    mcc = _mcc(TP, FP, FN, TN)
    aucs = _try_metrics(y_true_all, y_score_all)

    return {
        "meta": {
            "docs_processed": docs_processed,
            "docs_skipped_empty": docs_skipped_empty,
            "tokens_seen": tokens_seen_total,
            "expanded_size": len(det_terms),
            "ground_truth_size": len(gt_terms),
            "intersection_size": len(det_terms & gt_terms),
        },
        "counts": {"tp": TP, "fp": FP, "fn": FN, "tn": TN},
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "roc_auc_hard": aucs["roc_auc"],
            "average_precision_hard": aucs["avg_precision"],
            "note_auc": "AUCs computed with hard 0/1 scores; supply real-valued scores to make them more informative.",
        },
        "per_term": {
            "tp_terms": tp_terms.most_common(),
            "fp_terms": fp_terms.most_common(),
            "fn_terms": fn_terms.most_common(),
        },
    }

# ---------------------------
# CLI
# ---------------------------

def load_users(path: str | None) -> List[str] | None:
    with open(path, "r") as f:
        user_list = [user.strip() for user in f.read().split(",") if user.strip()]
    return user_list

def main():
    ap = argparse.ArgumentParser(description="Token-level evaluation of unigram expanded vs ground truth.")
    ap.add_argument("--expanded_json", required=True, help="Path to expanded JSON (dict of lists).")
    ap.add_argument("--ground_json",   required=True, help="Path to ground-truth JSON with key 'seed_terms'.")
    # return_documents parameters
    ap.add_argument("--db_name", default="reddit")
    ap.add_argument("--collection_name", default="noburp_all")
    ap.add_argument("--users", required=False)
    ap.add_argument("--out", required=True, help="Where to write the evaluation JSON.")
    args = ap.parse_args()

    exp_obj = load_json(args.expanded_json)
    gt_obj  = load_json(args.ground_json)

    exp_terms = _flatten_expanded(exp_obj)
    gt_terms  = _flatten_ground_truth(gt_obj)

    print(f"[info] expanded unigrams (normalized): {len(exp_terms)}")
    print(f"[info] ground-truth unigrams (normalized): {len(gt_terms)}")
    print(f"[info] intersection: {len(exp_terms & gt_terms)}")

    users = load_users(args.users)

    docs_raw = return_documents(
        db_name=args.db_name,
        collection_name=args.collection_name,
        filter_users=users,
    )

    if not isinstance(docs_raw, list) or (docs_raw and not isinstance(docs_raw[0], str)):
        raise TypeError("return_documents must return a List[str].")

    results = evaluate_token_level(docs_raw, exp_terms, gt_terms)

    out_path = Path(args.expanded_json).parent / Path(args.out).name
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")
    m = results["metrics"]; c = results["counts"]
    print(f"[done] tokens={results['meta']['tokens_seen']}  tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
    print(f"       P/R/F1={m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f}  Acc={m['accuracy']:.3f}  MCC={m['mcc']:.3f}")

if __name__ == "__main__":
    main()
