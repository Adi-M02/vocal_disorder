#!/usr/bin/env python3
"""
Analyze Word2Vec expansions with per-seed cosine stats and global overlap.

Usage:
  python analyze_expansions_with_stats.py \
      --model path/to/model.model \
      --seed_json path/to/seeds.json \
      [--topk 20]

Output:
  JSON file alongside the model, named like:
    <timestamp>_expansion<arch>/analysis_topk_<k>.json

Structure of the JSON:
{
  "meta": {...},
  "global_stats": {
      "cosine": {"mean":..., "median":..., "stdev":..., "min":..., "max":..., "p25":..., "p75":...},
      "hub_terms_top": [{"term":..., "count":..., "mean_cos":..., "seeds": [...] }]
  },
  "seeds": {
      "<seed>": {
          "seed_in_vocab": bool,
          "expansions": [{"term": str, "cos": float, "rank": int, "count": int|null, "hub_count": int, "is_seed": bool}],
          "avg_cos": float, "median_cos": float, "stdev_cos": float,
          "min_cos": float, "max_cos": float,
          "slope_1_to_k": float|null, "top1_gap": float|null, "tail_gap": float|null,
          "shared_with": [{"seed": str, "overlap": int, "jaccard": float}]
      }, ...
  }
}

Notes:
- Expansions are sorted by cosine similarity descending.
- "hub_count" is how many different seeds the term appears under (helps spot generic terms).
- "count" is the model's vocab count if available; otherwise null.
"""

import argparse, json, os, sys, re, statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
from gensim.models import Word2Vec

sys.path.append("../vocal_disorder")
from utils.text_pipeline import process_text
from utils.load_json import load_json

# ───────── JSON serialization helper (handles numpy types) ─────────
# Prevents: TypeError: Object of type int64 is not JSON serializable
# by converting numpy scalars/arrays to native Python types.
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


# ────────────────────────── helpers ──────────────────────────

def get_arch(model_path: Path, model: Word2Vec) -> str:
    m = re.search(r'_(cbow|skipgram)\.model$', model_path.name)
    if m:
        return m.group(1)
    return "skipgram" if getattr(model, "sg", 0) == 1 else "cbow"


def build_output_path(model_dir: Path, arch: str, k: int) -> Path:
    ts   = datetime.now().strftime("%m_%d_%H_%M")
    outd = model_dir / f"{ts}_expansion{arch}"
    outd.mkdir(parents=True, exist_ok=True)
    return outd / f"analysis_topk_{k}.json"


def seed_list_from_json(path: str) -> List[str]:
    obj = load_json(path)
    if isinstance(obj, dict):
        seeds = [t for lst in obj.values() for t in lst]
    else:
        seeds = list(obj)
    # normalize seeds to match model tokens
    return [" ".join(process_text(t)) for t in seeds]


def safe_get_count(model: Word2Vec, term: str):
    try:
        val = model.wv.get_vecattr(term, 'count')
        # Cast numpy int64 -> Python int for JSON safety
        return int(val) if val is not None else None
    except Exception:
        return None


def expansions_with_cos(model: Word2Vec, seed: str, k: int) -> List[Tuple[str, float]]:
    if seed not in model.wv:
        return []
    # gensim returns (term, cosine) already sorted desc
    return model.wv.most_similar(seed, topn=k)


def desc_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: None for k in ["mean","median","stdev","min","max","p25","p75"]}
    arr = np.array(values, dtype=float)
    out = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "stdev": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }
    return out


def analyze_seed(model: Word2Vec, seed: str, k: int, seed_set: set) -> Dict[str, Any]:
    exps = expansions_with_cos(model, seed, k)
    cos_vals = [c for _, c in exps]
    # micro-stats
    stats = desc_stats(cos_vals)
    top1_gap = (cos_vals[0] - cos_vals[1]) if len(cos_vals) >= 2 else None
    tail_gap = (cos_vals[-2] - cos_vals[-1]) if len(cos_vals) >= 2 else None
    slope = (cos_vals[0] - cos_vals[-1]) if len(cos_vals) >= 2 else None

    # assemble expansion dicts (hub_count filled later)
    expansions = [
        {
            "term": t,
            "cos": float(c),
            "rank": int(i + 1),
            "count": safe_get_count(model, t),
            "hub_count": None,   # placeholder; filled post global pass
            "is_seed": (t in seed_set),
        }
        for i, (t, c) in enumerate(exps)
    ]

    return {
        "seed_in_vocab": bool(seed in model.wv),
        "expansions": expansions,
        "avg_cos": stats["mean"],
        "median_cos": stats["median"],
        "stdev_cos": stats["stdev"],
        "min_cos": stats["min"],
        "max_cos": stats["max"],
        "slope_1_to_k": slope,
        "top1_gap": top1_gap,
        "tail_gap": tail_gap,
        # filled later: shared_with
    }


def compute_overlap(seeds_to_terms: Dict[str, List[str]], top_n: int = 5):
    # Per-seed top overlaps with other seeds
    seed_names = list(seeds_to_terms.keys())
    term_sets = {s: set(tlist) for s, tlist in seeds_to_terms.items()}
    result = {s: [] for s in seed_names}
    for i, s1 in enumerate(seed_names):
        for j in range(i+1, len(seed_names)):
            s2 = seed_names[j]
            inter = term_sets[s1] & term_sets[s2]
            if not inter:
                continue
            union = term_sets[s1] | term_sets[s2]
            jacc = len(inter) / len(union)
            result[s1].append((s2, len(inter), jacc))
            result[s2].append((s1, len(inter), jacc))
    # keep top-N per seed by overlap size then jaccard
    top = {}
    for s, lst in result.items():
        lst.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top[s] = [
            {"seed": other, "overlap": ov, "jaccard": round(jc, 4)}
            for other, ov, jc in lst[:top_n]
        ]
    return top


# ────────────────────────── main ──────────────────────────

def main(args):
    model_path = Path(args.model).expanduser()
    model = Word2Vec.load(str(model_path))
    arch = get_arch(model_path, model)

    seed_terms = seed_list_from_json(args.seed_json)
    seed_set = set(seed_terms)

    # per-seed analyses
    seeds_out: Dict[str, Any] = {}
    all_cos: List[float] = []
    term_hub_cos: Dict[str, List[float]] = {}
    term_hub_seeds: Dict[str, List[str]] = {}

    for s in seed_terms:
        entry = analyze_seed(model, s, args.topk, seed_set)
        seeds_out[s] = entry
        # aggregate for global stats
        for d in entry["expansions"]:
            all_cos.append(d["cos"])
            term = d["term"]
            term_hub_cos.setdefault(term, []).append(d["cos"])
            term_hub_seeds.setdefault(term, []).append(s)

    # compute hub counts and annotate expansions
    hub_counts = {t: len(set(seeds)) for t, seeds in term_hub_seeds.items()}
    for s, entry in seeds_out.items():
        for d in entry["expansions"]:
            d["hub_count"] = hub_counts.get(d["term"], 1)

    # top hubs
    hub_items = []
    for t, seeds in term_hub_seeds.items():
        hub_items.append({
            "term": t,
            "count": len(set(seeds)),
            "mean_cos": float(np.mean(term_hub_cos[t])) if term_hub_cos[t] else None,
            "seeds": sorted(list(set(seeds)))[:10],  # truncate to first 10 for brevity
        })
    hub_items.sort(key=lambda x: (x["count"], x["mean_cos"] if x["mean_cos"] is not None else -1.0), reverse=True)

    # pairwise overlaps (top5 per seed)
    seeds_to_terms = {s: [d["term"] for d in entry["expansions"]] for s, entry in seeds_out.items()}
    overlaps = compute_overlap(seeds_to_terms, top_n=5)
    for s in seed_terms:
        seeds_out[s]["shared_with"] = overlaps.get(s, [])

    # global cosine stats
    global_cos_stats = desc_stats(all_cos)

    # assemble final JSON
    out = {
        "meta": {
            "model": str(model_path),
            "arch": arch,
            "k": int(args.topk),
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "num_seeds": len(seed_terms),
            "version": 1
        },
        "global_stats": {
            "cosine": global_cos_stats,
            "hub_terms_top": hub_items[:50],  # top 50 hubs
        },
        "seeds": seeds_out,
    }

    out_path = build_output_path(model_path.parent, arch, args.topk)
    out_path.write_text(json.dumps(out, indent=2, default=_to_jsonable))

    # console summary
    print(f"Processed {len(seed_terms)} seeds from {model_path.name} (arch={arch}, k={args.topk})")
    print(f"→ Wrote {out_path}")

    # Show a few quick insights
    if all_cos:
        print("Global cosine stats:")
        print(json.dumps(global_cos_stats, indent=2))

    # seeds with lowest avg cosine (possible weak neighborhoods)
    by_avg = [
        (s, seeds_out[s]["avg_cos"] if seeds_out[s]["avg_cos"] is not None else -999)
        for s in seed_terms
    ]
    by_avg = [x for x in by_avg if x[1] != -999]
    if by_avg:
        by_avg.sort(key=lambda x: x[1])
        worst = by_avg[:5]
        print("\nSeeds with lowest avg cosine:")
        for s, v in worst:
            print(f"  {s:>20}  avg_cos={v:.4f}  top1={seeds_out[s]['expansions'][0]['cos']:.4f}  last={seeds_out[s]['expansions'][-1]['cos']:.4f}")

    # top hub terms
    if hub_items:
        print("\nTop hub expansion terms (appear under many seeds):")
        for h in hub_items[:10]:
            print(f"  {h['term']:<25} count={h['count']:<3} mean_cos={h['mean_cos']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze nearest-neighbor expansions with cosine stats and overlaps")
    p.add_argument("--model", required=True, help="Path to Word2Vec .model file")
    p.add_argument("--seed_json", required=True, help="JSON list or {cat: [terms]}")
    p.add_argument("--topk", type=int, default=20, help="Top-k neighbors per seed")
    args = p.parse_args()
    main(args)
