"""
End-to-end evaluation of Word2Vec expansions on corpus tokens with seed attribution.

Usage
-----
python expand_and_evaluate_with_attribution.py \
  --model path/to/model.model \
  --seed_json path/to/seeds.json \
  --ground_json path/to/ground.json \
  --db_name reddit \
  --collection_name noburp_all \
  --users users.txt \                # optional CSV: u1,u2,...
  --topk 20 \
  [--credit_mode fractional|duplicate] \
  [--include_seeds]                  # default OFF to match your pipeline
"""

import argparse, collections, json, os, re, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set, Optional

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, average_precision_score

# Project imports
sys.path.append("../vocal_disorder")
from testing.test_ngram_generation import load_phrasers_from_dir, apply_ngrams
from utils.text_pipeline import process_text, remove_unigram_stopwords
from utils.load_json import load_json
from query_mongo import return_documents


# ───────────────────── Helpers copied/adapted from your scripts ─────────────────────

def get_arch_from_name_or_attr(model_path: Path, model: Word2Vec) -> str:
    """
    Prefer model filename: “…_cbow.model” / “…_skipgram.model”.
    Fallback to gensim's model.sg (0=cbow, 1=skipgram).
    """
    m = re.search(r'_(cbow|skipgram)\.model$', model_path.name)
    if m:
        return m.group(1)
    return "skipgram" if getattr(model, "sg", 0) == 1 else "cbow"

def most_similar_exact(model: Word2Vec, term: str, k: int, min_cos: Optional[float]) -> List[str]:
    """
    EXACT behavior of your expander:
      - get topk neighbors
      - if min_cos provided, keep only neighbors with cosine > min_cos (strict)
      - return only neighbor TERMS (no cosine)
    """
    if term not in model.wv:
        return []
    sims = model.wv.most_similar(positive=[term], topn=k)
    if min_cos is None:
        return [t for t, _ in sims]
    return [t for t, cos in sims if cos > float(min_cos)]

def load_ground_unigrams(ground_json_path: str) -> Set[str]:
    obj = load_json(ground_json_path)
    if not (isinstance(obj, dict) and "seed_terms" in obj and isinstance(obj["seed_terms"], list)):
        raise ValueError("Ground truth JSON must be an object with a 'seed_terms' list.")
    return set(obj["seed_terms"])

def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0

def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return ((tp*tn - fp*fn) / (denom ** 0.5)) if denom else 0.0

def _try_metrics_hard(y_true: List[int], y_score: List[int]) -> Dict[str, Optional[float]]:
    out = {"roc_auc_hard": None, "average_precision_hard": None}
    if not y_true:
        return out
    classes = set(y_true)
    if classes == {0, 1}:
        try:
            out["roc_auc_hard"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    if 1 in classes:
        try:
            out["average_precision_hard"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass
    return out

def _load_users(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [u.strip() for u in f.read().split(",") if u.strip()]

def _to_jsonable(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    return o


# ───────────────────────────── Expansion (exact) ─────────────────────────────

def expand_seeds_exact(model: Word2Vec, seed_json: str, topk: int, min_cos: Optional[float]):
    seeds_obj = load_json(seed_json)

    # Build seeds list and optional category map (if input is {cat: [terms]})
    seed_to_cat: Optional[Dict[str, str]] = None
    if isinstance(seeds_obj, dict):
        pairs = [(t, cat) for cat, lst in seeds_obj.items() for t in lst]
        seeds = [t for t, _ in pairs]
        seed_to_cat = {}
        for (t, c), s_norm in zip(pairs, seeds):
            if s_norm:
                seed_to_cat[s_norm] = c
    else:
        seeds = list(seeds_obj)

    expansions: Dict[str, List[str]] = {}
    for s in seeds:
        if not s:
            expansions[s] = []
            continue
        expansions[s] = most_similar_exact(model, s, topk, min_cos)

    return seeds, expansions, seed_to_cat


# ───────────────────────────── Evaluation (exact) ─────────────────────────────

def evaluate_on_docs_exact(
    docs_raw: List[str],
    det_terms: Set[str],
    gt_terms: Set[str],
    ngram_phraser_dir: str
):
    """Exact clone of your evaluator’s token-level logic with hard scores."""
    import collections as _collections
    TP = FP = FN = TN = 0
    tp_terms = _collections.Counter()
    fp_terms = _collections.Counter()
    fn_terms = _collections.Counter()

    y_true_all: List[int] = []
    y_score_all: List[int] = []

    tokens_seen_total = 0
    docs_processed = 0
    docs_skipped_empty = 0
    bigram, trigram = load_phrasers_from_dir(ngram_phraser_dir)
    for text in docs_raw:
        docs_processed += 1
        if not isinstance(text, str) or not text:
            docs_skipped_empty += 1
            continue

        toks = process_text(text, stoplist=False)
        if not toks:
            docs_skipped_empty += 1
            continue
        toks = apply_ngrams(toks, (bigram, trigram))
        toks = remove_unigram_stopwords(toks)

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
    aucs = _try_metrics_hard(y_true_all, y_score_all)

    results = {
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
            **aucs,
            "note_auc": "AUCs computed with hard 0/1 scores (matches original evaluator).",
        },
        "per_term": {
            "tp_terms": tp_terms.most_common(),
            "fp_terms": fp_terms.most_common(),
            "fn_terms": fn_terms.most_common(),
        },
    }
    return results


# ───────────────────── Attribution & extra analyses (do NOT change metrics) ─────────────────────

def build_term_to_seeds(expansions: Dict[str, List[str]], include_seeds: bool, seeds: List[str]) -> Dict[str, Set[str]]:
    term_to_seeds: Dict[str, Set[str]] = {}
    for s, lst in expansions.items():
        for t in lst:
            term_to_seeds.setdefault(t, set()).add(s)
    if include_seeds:
        for s in seeds:
            term_to_seeds.setdefault(s, set()).add(s)
    return term_to_seeds

def attribute_tp_fp_tokens(
    docs_raw: List[str],
    gt_terms: Set[str],
    det_terms: Set[str],
    term_to_seeds: Dict[str, Set[str]],
    credit_mode: str = "fractional",  # or "duplicate"
    seed_to_cat: Optional[Dict[str, str]] = None,
    ngram_phraser_dir: str = None
):
    seed_attr: Dict[str, Any] = {}
    cat_attr: Optional[Dict[str, Any]] = {} if seed_to_cat else None

    def _init_seed(s):
        if s not in seed_attr:
            seed_attr[s] = {
                "tp_tokens": 0.0,
                "fp_tokens": 0.0,
                "precision_tokens": None,
                "top_tp_terms": collections.Counter(),
                "top_fp_terms": collections.Counter(),
            }

    def _init_cat(cat):
        if cat_attr is None:
            return
        if cat not in cat_attr:
            cat_attr[cat] = {
                "tp_tokens": 0.0,
                "fp_tokens": 0.0,
                "precision_tokens": None,
                "top_tp_terms": collections.Counter(),
                "top_fp_terms": collections.Counter(),
            }
    bigram, trigram = load_phrasers_from_dir(ngram_phraser_dir)
    for text in docs_raw:
        if not isinstance(text, str) or not text:
            continue
        toks = process_text(text, stoplist=False)
        if not toks:
            continue
        toks = apply_ngrams(toks, (bigram, trigram))
        toks = remove_unigram_stopwords(toks)
        for tok in toks:
            is_gt = tok in gt_terms
            is_det = tok in det_terms
            if not is_det:
                continue
            seeds = term_to_seeds.get(tok, set())
            if not seeds:
                continue
            w = (1.0/len(seeds)) if credit_mode == "fractional" else 1.0
            if is_gt:
                for s in seeds:
                    _init_seed(s)
                    seed_attr[s]["tp_tokens"] += w
                    seed_attr[s]["top_tp_terms"][tok] += w
                    if seed_to_cat and s in seed_to_cat:
                        c = seed_to_cat[s]; _init_cat(c)
                        cat_attr[c]["tp_tokens"] += w
                        cat_attr[c]["top_tp_terms"][tok] += w
            else:
                for s in seeds:
                    _init_seed(s)
                    seed_attr[s]["fp_tokens"] += w
                    seed_attr[s]["top_fp_terms"][tok] += w
                    if seed_to_cat and s in seed_to_cat:
                        c = seed_to_cat[s]; _init_cat(c)
                        cat_attr[c]["fp_tokens"] += w
                        cat_attr[c]["top_fp_terms"][tok] += w

    for s, d in seed_attr.items():
        denom = d["tp_tokens"] + d["fp_tokens"]
        d["precision_tokens"] = (d["tp_tokens"] / denom) if denom else None
        d["top_tp_terms"] = d["top_tp_terms"].most_common(25)
        d["top_fp_terms"] = d["top_fp_terms"].most_common(25)
    if cat_attr is not None:
        for c, d in cat_attr.items():
            denom = d["tp_tokens"] + d["fp_tokens"]
            d["precision_tokens"] = (d["tp_tokens"] / denom) if denom else None
            d["top_tp_terms"] = d["top_tp_terms"].most_common(25)
            d["top_fp_terms"] = d["top_fp_terms"].most_common(25)

    return seed_attr, cat_attr

def compute_hubs_and_overlaps(expansions: Dict[str, List[str]]):
    # hubs: terms that appear under many seeds
    term_to_seeds = {}
    for s, lst in expansions.items():
        for t in lst:
            term_to_seeds.setdefault(t, set()).add(s)
    hubs = [{"term": t, "hub_count": len(seeds), "seeds": sorted(list(seeds))[:10]}
            for t, seeds in term_to_seeds.items()]
    hubs.sort(key=lambda x: x["hub_count"], reverse=True)

    # overlaps: per-seed overlap with others (top 5)
    seeds_to_sets = {s: set(lst) for s, lst in expansions.items()}
    overlaps: Dict[str, List[Dict[str, Any]]] = {s: [] for s in expansions.keys()}
    names = list(expansions.keys())
    for i, s1 in enumerate(names):
        for j in range(i+1, len(names)):
            s2 = names[j]
            a, b = seeds_to_sets[s1], seeds_to_sets[s2]
            inter = a & b
            if not inter:
                continue
            jacc = len(inter) / len(a | b)
            overlaps[s1].append({"seed": s2, "overlap": len(inter), "jaccard": round(jacc, 4)})
            overlaps[s2].append({"seed": s1, "overlap": len(inter), "jaccard": round(jacc, 4)})
    overlaps = {s: sorted(lst, key=lambda x: (x["overlap"], x["jaccard"]), reverse=True)[:5]
                for s, lst in overlaps.items()}
    return hubs[:50], overlaps


# ───────────────────────────── Output paths ─────────────────────────────

def build_output_dir(model_path: Path, arch: str) -> Path:
    ts = datetime.now().strftime("%m_%d_%H_%M")
    out_dir = model_path.parent / f"{arch}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def build_file_paths(out_dir: Path, topk: int, min_cos: Optional[float]) -> Tuple[Path, Path]:
    suffix = f"_min_cos_{min_cos:g}" if min_cos is not None else ""
    exp_path = out_dir / f"expanded_topk_{topk}{suffix}.json"
    ana_path = out_dir / f"analysis_topk_{topk}{suffix}.json"
    return exp_path, ana_path


# ─────────────────────────────────── Main ───────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Exact expansion + exact evaluation + seed attribution (writes 2 JSON files).")
    # Expansion args (exact)
    p.add_argument("--model", required=True, help="Path to Word2Vec .model")
    p.add_argument("--seed_json", required=True, help="Seeds JSON (list or {category: [terms]})")
    p.add_argument("--topk", type=int, default=25, help="Top-k neighbors per seed")
    p.add_argument("--min_cos", type=float, default=0.4
    , help="Min cosine; strict '>' filter after topk")
    # Evaluation args (exact)
    p.add_argument("--ground_json", required=True, help="Ground JSON with key 'seed_terms' (unigrams after normalization)")
    p.add_argument("--db_name", default="reddit")
    p.add_argument("--collection_name", default="noburp_all")
    p.add_argument("--users", default="vocabulary_evaluation/manual_terms_7_12/users.txt", help="Optional CSV of users: u1,u2,...")
    # Extras
    p.add_argument("--credit_mode", choices=["fractional","duplicate"], default="duplicate")
    p.add_argument("--include_seeds", action="store_true", help="Include original seeds as detector terms (OFF by default to match original)")
    p.add_argument("--ngram_phraser_dir", type=str, required=True, help="Directory containing n-gram phrasers.")
    args = p.parse_args()

    # Load model
    model_path = Path(args.model).expanduser()
    model = Word2Vec.load(str(model_path))
    arch = get_arch_from_name_or_attr(model_path, model)

    # 1) Expand (EXACT)
    seeds_norm, expansions, seed_to_cat = expand_seeds_exact(model, args.seed_json, args.topk, args.min_cos)

    # detector terms (EXACT): union of all expanded lists
    det_terms = set([t for lst in expansions.values() for t in lst])
    if args.include_seeds:
        det_terms |= set(seeds_norm)  # optional; off by default

    # 2) Ground set (normalized unigrams)
    gt_terms = load_ground_unigrams(args.ground_json)

    # 3) Pull docs
    users = _load_users(args.users)
    docs_raw = return_documents(
        db_name=args.db_name,
        collection_name=args.collection_name,
        filter_users=users,
    )
    if not isinstance(docs_raw, list) or (docs_raw and not isinstance(docs_raw[0], str)):
        raise TypeError("return_documents must return a List[str].")

    # 4) Evaluate (EXACT)
    results_eval = evaluate_on_docs_exact(docs_raw, det_terms, gt_terms, args.ngram_phraser_dir)

    # 5) Attribution & extras (do NOT change metrics)
    term_to_seeds = build_term_to_seeds(expansions, args.include_seeds, seeds_norm)
    seed_attr, cat_attr = attribute_tp_fp_tokens(
        docs_raw=docs_raw,
        gt_terms=gt_terms,
        det_terms=det_terms,
        term_to_seeds=term_to_seeds,
        credit_mode=args.credit_mode,
        seed_to_cat=seed_to_cat,
        ngram_phraser_dir=args.ngram_phraser_dir
    )
    hubs_top, overlaps = compute_hubs_and_overlaps(expansions)

    # 6) Build output dir + file paths
    out_dir = build_output_dir(model_path, arch)
    exp_json_path, analysis_json_path = build_file_paths(out_dir, args.topk, args.min_cos)

    # 7) Write PLAIN expanded JSON (exactly like your first script)
    #    Format: {seed: [neighbor_terms]}
    exp_json_path.write_text(json.dumps(expansions, indent=2), encoding="utf-8")

    # 8) Write analysis/evaluation JSON (extra info)
    out_obj = {
        "meta": {
            "model": str(model_path),
            "ground_json": args.ground_json,
            "arch": arch,
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "topk": int(args.topk),
            "min_cos": args.min_cos,
            "include_seeds": bool(args.include_seeds),
            "credit_mode": args.credit_mode,
            "num_seeds": len(seeds_norm),
            "output_dir": str(out_dir),
        },
        "expansion": {
            "by_seed": expansions,            # kept for convenience
            "detector_size": len(det_terms),
            "hub_terms_top": hubs_top,
            "overlaps_top5_per_seed": overlaps,
        },
        "evaluation": results_eval,           # EXACT metrics & per-term counts
        "attribution": {
            "per_seed": seed_attr,
            "per_category": cat_attr,
        },
    }
    analysis_json_path.write_text(json.dumps(out_obj, indent=2, default=_to_jsonable), encoding="utf-8")

    # Console summary
    c = results_eval["counts"]; m = results_eval["metrics"]
    print(f"[saved] expanded ➜ {exp_json_path}")
    print(f"[saved] analysis ➜ {analysis_json_path}")
    print(f"[done] tokens={results_eval['meta']['tokens_seen']}  tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
    print(f"       P/R/F1={m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f}  Acc={m['accuracy']:.3f}  MCC={m['mcc']:.3f}")
    if m.get("roc_auc_hard") is not None or m.get("average_precision_hard") is not None:
        print(f"       ROC_AUC(hard)={m.get('roc_auc_hard')}  PR_AUC(hard)={m.get('average_precision_hard')}")

if __name__ == "__main__":
    main()
