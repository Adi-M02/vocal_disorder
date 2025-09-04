#!/usr/bin/env python3
"""
Compare filtering methods over seed expansions.

Given:
  1) Base expansions         : { "<seed>": ["cand1","cand2",...], ... }
  2) LLM accepted-by-seed    : { "<seed>": ["accepted1",...], ... }
  3) WordNet accepted-by-seed: { "<seed>": ["accepted1",...], ... }

Produces, under --outdir/compare_<timestamp>/ :
  - per_seed_diffs.json  (readable per-seed lists)
  - per_seed_summary.csv (seed-level counts + Jaccard)
  - global_summary.json  (aggregate counts + sets sizes)

Usage:
  python compare_filter_methods.py \
      --base path/to/base_expansions.json \
      --llm path/to/llm_accepted.json \
      --wordnet path/to/wordnet_accepted_by_seed.json \
      --outdir path/to/outdir \
      [--top 25]
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm_key(s: str) -> str:
    return s.strip().lower()

def norm_list(xs):
    return sorted({(x or "").strip().lower() for x in xs if isinstance(x, str)})

def main():
    ap = argparse.ArgumentParser(description="Compare LLM vs WordNet filtering outputs.")
    ap.add_argument("--base", required=True, help="Base expansions JSON {seed: [cands]}")
    ap.add_argument("--llm", required=True, help="LLM accepted-by-seed JSON {seed: [accepted]}")
    ap.add_argument("--wordnet", required=True, help="WordNet accepted-by-seed JSON {seed: [accepted]}")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--top", type=int, default=25, help="How many top disagreements to print")
    args = ap.parse_args()

    base_path = Path(args.base).expanduser().resolve()
    llm_path = Path(args.llm).expanduser().resolve()
    wn_path = Path(args.wordnet).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    stamp = datetime.now().strftime("%m_%d_%H_%M")
    outdir = outdir / f"compare_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_json(base_path)
    llm = load_json(llm_path)
    wn  = load_json(wn_path)

    # Normalize keys and values
    # Seeds universe prioritizes base.keys() but we will union with llm/wn to catch stray seeds.
    seeds: Set[str] = set()
    base_n: Dict[str, List[str]] = {}
    for k, v in (base.items() if isinstance(base, dict) else []):
        kn = norm_key(k); seeds.add(kn)
        base_n[kn] = norm_list(v if isinstance(v, list) else [])

    llm_n: Dict[str, List[str]] = {}
    for k, v in (llm.items() if isinstance(llm, dict) else []):
        kn = norm_key(k); seeds.add(kn)
        llm_n[kn] = norm_list(v if isinstance(v, list) else [])

    wn_n: Dict[str, List[str]] = {}
    for k, v in (wn.items() if isinstance(wn, dict) else []):
        kn = norm_key(k); seeds.add(kn)
        wn_n[kn] = norm_list(v if isinstance(v, list) else [])

    # Build per-seed diffs
    per_seed_diffs: Dict[str, dict] = {}
    summary_rows: List[Tuple] = []

    # Global aggregates
    global_unique_base: Set[str] = set()
    global_unique_llm: Set[str] = set()
    global_unique_wn: Set[str] = set()
    global_only_llm: Set[str] = set()
    global_only_wn: Set[str] = set()
    global_removed_by_both: Set[str] = set()
    global_llm_extra: Set[str] = set()
    global_wn_extra: Set[str] = set()

    for seed in sorted(seeds):
        base_terms = set(base_n.get(seed, []))
        llm_keep   = set(llm_n.get(seed, []))
        wn_keep    = set(wn_n.get(seed, []))

        # extras (should be rare): in accepted but not in base
        llm_extra = llm_keep - base_terms
        wn_extra  = wn_keep - base_terms

        both      = llm_keep & wn_keep
        only_llm  = llm_keep - wn_keep
        only_wn   = wn_keep - llm_keep

        removed_by_llm  = base_terms - llm_keep
        removed_by_wn   = base_terms - wn_keep
        removed_by_both = base_terms - (llm_keep | wn_keep)

        # Jaccard similarity between keeps (handle 0/0 gracefully)
        denom = len(llm_keep | wn_keep)
        jaccard = (len(both) / denom) if denom else 1.0

        per_seed_diffs[seed] = {
            "base_terms":           sorted(base_terms),
            "llm_keep":             sorted(llm_keep),
            "wordnet_keep":         sorted(wn_keep),
            "intersection":         sorted(both),
            "only_llm":             sorted(only_llm),
            "only_wordnet":         sorted(only_wn),
            "removed_by_llm":       sorted(removed_by_llm),
            "removed_by_wordnet":   sorted(removed_by_wn),
            "removed_by_both":      sorted(removed_by_both),
            "llm_extra_not_in_base": sorted(llm_extra),
            "wordnet_extra_not_in_base": sorted(wn_extra),
            "jaccard":              jaccard
        }

        summary_rows.append((
            seed,
            len(base_terms),
            len(llm_keep),
            len(wn_keep),
            len(both),
            len(only_llm),
            len(only_wn),
            len(removed_by_both),
            f"{jaccard:.4f}"
        ))

        # accumulate global sets
        global_unique_base |= base_terms
        global_unique_llm  |= llm_keep
        global_unique_wn   |= wn_keep
        global_only_llm    |= only_llm
        global_only_wn     |= only_wn
        global_removed_by_both |= removed_by_both
        global_llm_extra   |= llm_extra
        global_wn_extra    |= wn_extra

    # Write per-seed JSON diffs
    with open(outdir / "per_seed_diffs.json", "w", encoding="utf-8") as f:
        json.dump(per_seed_diffs, f, indent=2, ensure_ascii=False)

    # Write per-seed CSV summary
    with open(outdir / "per_seed_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seed","base_n","llm_n","wordnet_n",
            "intersection_n","only_llm_n","only_wordnet_n",
            "removed_by_both_n","jaccard"
        ])
        for row in summary_rows:
            w.writerow(row)

    # Global summary
    global_summary = {
        "num_seeds": len(seeds),
        "unique_base_terms": len(global_unique_base),
        "unique_llm_kept_terms": len(global_unique_llm),
        "unique_wordnet_kept_terms": len(global_unique_wn),
        "unique_only_llm_terms": len(global_only_llm),
        "unique_only_wordnet_terms": len(global_only_wn),
        "unique_removed_by_both_terms": len(global_removed_by_both),
        "unique_llm_extra_not_in_base": len(global_llm_extra),
        "unique_wordnet_extra_not_in_base": len(global_wn_extra),
        "paths": {
            "base": str(base_path),
            "llm": str(llm_path),
            "wordnet": str(wn_path)
        }
    }
    with open(outdir / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    # Print quick console preview (top disagreements)
    by_disagree = sorted(
        summary_rows,
        key=lambda r: (r[5] + r[6], -float(r[8])),  # more one-sided keeps; tie-break by lower Jaccard
        reverse=True
    )
    top = max(0, int(args.top))
    print(f"\n=== Global Summary ===")
    print(json.dumps(global_summary, indent=2))
    print(f"\n=== Top {top} seeds by disagreement (only_llm_n + only_wordnet_n) ===")
    for seed, base_n, llm_n, wn_n, inter_n, only_llm_n, only_wn_n, rem_both_n, jacc in by_disagree[:top]:
        print(f"- {seed}: base={base_n}, llm={llm_n}, wn={wn_n}, "
              f"∩={inter_n}, only_llm={only_llm_n}, only_wn={only_wn_n}, "
              f"removed_by_both={rem_both_n}, jaccard={jacc}")

if __name__ == "__main__":
    main()