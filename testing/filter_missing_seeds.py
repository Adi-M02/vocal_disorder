#!/usr/bin/env python3
"""
Filter seed terms that are NOT present as either 'seed' or 'candidate' in an NDJSON file.

Usage:
  python find_missing_terms.py --json seeds.json --ndjson decisions.ndjson
  python find_missing_terms.py --json seeds.json --ndjson decisions.ndjson --out missing.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_seed_terms(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "seed_terms" in data:
        terms = data["seed_terms"]
    elif isinstance(data, list):
        terms = data
    else:
        raise ValueError("Input JSON must be an object with 'seed_terms' or a list of strings.")
    if not all(isinstance(t, str) for t in terms):
        raise ValueError("All seed terms must be strings.")
    return terms


def load_ndjson_terms(path: Path):
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed/truncated lines
                print(f"warn: skipping malformed JSON at line {i}", file=sys.stderr)
                continue
            for key in ("seed", "candidate"):
                val = obj.get(key)
                if isinstance(val, str):
                    seen.add(val)
    return seen


def main():
    ap = argparse.ArgumentParser(
        description="Return seed terms from a JSON that do not appear as 'seed' or 'candidate' in an NDJSON."
    )
    ap.add_argument("--json", required=True, type=Path, help="Path to JSON with 'seed_terms' array.")
    ap.add_argument("--ndjson", required=True, type=Path, help="Path to NDJSON with 'seed' and 'candidate' fields.")
    ap.add_argument("--out", type=Path, help="Optional output path to write the missing terms as a JSON array.")
    args = ap.parse_args()

    seed_terms = load_seed_terms(args.json)
    ndjson_seen = load_ndjson_terms(args.ndjson)

    missing = [t for t in seed_terms if t not in ndjson_seen]

    if args.out:
        args.out.write_text(json.dumps(missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(missing)} terms to {args.out}")
    else:
        json.dump(missing, sys.stdout, ensure_ascii=False, indent=2)
        print()  # newline


if __name__ == "__main__":
    main()
