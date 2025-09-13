#!/usr/bin/env python3
"""
Sample non-empty key/value pairs from a JSON dict.

Given a JSON file whose top-level is a dict[str, list], this script samples
N=25 (configurable) key/value pairs **only from keys whose values are
non-empty lists** and writes them to an output JSON file.

Usage:
  python sample_nonempty_pairs.py input.json \
      --count 25 \
      --seed 42 \
      --output sampled.json

If --output is omitted, the script writes to "<input>.sample_<count>.json".
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample non-empty list-valued pairs from a JSON dict.")
    p.add_argument("input", type=Path, help="Path to input JSON file (top-level dict).")
    p.add_argument("-c", "--count", type=int, default=25, help="Number of pairs to sample (default: 25).")
    p.add_argument("-s", "--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output JSON file path.")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        with args.input.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print("ERROR: Top-level JSON must be an object/dict.", file=sys.stderr)
        sys.exit(1)

    # Keep only items whose value is a non-empty list
    eligible_items = {k: v for k, v in data.items() if isinstance(v, list) and len(v) > 0}

    if len(eligible_items) == 0:
        print("WARNING: No keys with non-empty list values found. Output will be empty.", file=sys.stderr)
        sampled = {}
    else:
        # Stable ordering before sampling (helps reproducibility across platforms)
        eligible_keys = sorted(eligible_items.keys())

        rng = random.Random(args.seed)
        k = min(args.count, len(eligible_keys))
        if k < args.count:
            print(f"NOTE: Only {len(eligible_keys)} eligible keys; sampling {k}.", file=sys.stderr)

        sampled_keys = rng.sample(eligible_keys, k)
        sampled = {k: eligible_items[k] for k in sampled_keys}

    out_path = args.output
    if out_path is None:
        out_path = args.input.with_suffix(args.input.suffix + f".sample_{args.count}.json")

    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(sampled)} key/value pairs to {out_path}")

if __name__ == "__main__":
    main()
