#!/usr/bin/env python3
"""
Interactive reviewer (cached corpus, minimal)
- Excludes ONLY seed keys from review.
- Processes terms in the same order they appear in the input JSON (dedup by first occurrence).
- Shows up to 2 corpus contexts via query_base_windows (window=12).
- Prompt: [y]es/[n]o/[s]kip/[q]uit + required reasoning.
- CSV: term,decision,reasoning,context1,context2
"""

import argparse
import csv
import json
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import OrderedDict

import pandas as pd

sys.path.append("../vocal_disorder")
from testing.adding_context.context_api import load_cached_corpus, query_base_windows

# ---------- UI ----------
def start_spinner(msg: str):
    stop = {"flag": False}
    def run():
        frames = "|/-\\"
        i = 0
        while not stop["flag"]:
            print(f"\r{msg} {frames[i % len(frames)]}", end="", flush=True)
            i += 1
            time.sleep(0.1)
        print("\r" + " " * (len(msg) + 2) + "\r", end="", flush=True)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    def stop_fn():
        stop["flag"] = True
        th.join()
    return stop_fn

def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        return "q"

# ------- I/O -------
def load_expansions(path: Path) -> "OrderedDict[str, List[str]]":
    text = path.read_text(encoding="utf-8")
    data = json.loads(text, object_pairs_hook=OrderedDict)  # preserve seed order
    if not isinstance(data, dict):
        raise ValueError("Expansions JSON must map seeds -> list of terms.")
    fixed = OrderedDict()
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"Seed '{k}' must map to a list.")
        fixed[k] = [s for s in (str(x).strip() for x in v) if s]
    return fixed

def compute_candidates_ordered(expansions: "OrderedDict[str, List[str]]") -> Tuple[List[str], List[str]]:
    """Return (new_terms_in_order, excluded_terms_in_order); exclude = seeds only."""
    seeds: Set[str] = set(expansions.keys())
    seen: Set[str] = set()
    new_terms: List[str] = []
    excluded: List[str] = []
    for seed in expansions.keys():           # seed order
        for t in expansions[seed]:           # term order per seed
            if t in seen:
                continue
            seen.add(t)
            if t in seeds:
                excluded.append(t)
            else:
                new_terms.append(t)
    return new_terms, excluded

# ------ CSV ------
HEADERS = ["term", "decision", "reasoning", "context1", "context2"]

def ensure_csv_header(csv_path: Path) -> List[str]:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            first = next(csv.reader(f), None)
        if first:
            return first
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(HEADERS)
    return HEADERS

def load_existing_decisions(csv_path: Path) -> Set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    decided: Set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "term" not in reader.fieldnames:
            return set()
        for row in reader:
            t = (row.get("term") or "").strip()
            if t:
                decided.add(t)
    return decided

def append_row(csv_path: Path, record: dict):
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        ctx = record.get("contexts") or []
        w.writerow({
            "term": record.get("term", ""),
            "decision": record.get("decision", ""),
            "reasoning": record.get("reasoning", ""),
            "context1": ctx[0] if len(ctx) > 0 else "",
            "context2": ctx[1] if len(ctx) > 1 else "",
        })

# ---- Context formatting ----
def extract_context_strings(windows_per_doc: List[List[str]], max_n: int = 2) -> List[str]:
    """API returns like: [['...'], ['...']]. Take first string per doc, dedup, cap at max_n."""
    out: List[str] = []
    seen: Set[str] = set()
    for per_doc in windows_per_doc:
        if not per_doc:
            continue
        s = per_doc[0]
        s = s.strip() if isinstance(s, str) else str(s)
        if s and s not in seen:
            out.append(s)
            seen.add(s)
            if len(out) >= max_n:
                break
    return out

# ------- Main -------
def main():
    ap = argparse.ArgumentParser(description="Interactive manual eval with cached corpus contexts.")
    ap.add_argument("--expansions", required=True, type=Path, help="Path to expansions JSON.")
    ap.add_argument("--cache", default="base_ngram_cache_with_details.parquet", type=str, help="Cached corpus path for load_cached_corpus.")
    ap.add_argument("--cache-format", choices=["parquet", "jsonl"], default="parquet")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV (append-only).")
    ap.add_argument("--resume", action="store_true", help="Skip terms already decided.")
    args = ap.parse_args()

    expansions = load_expansions(args.expansions)
    new_terms, excluded_terms = compute_candidates_ordered(expansions)  # preserves JSON order

    print(f"Seeds (excluded from review): {len(expansions)}")
    print(f"New unique candidates to review: {len(new_terms)}")
    print(f"Excluded candidates (seed overlap): {len(excluded_terms)}")

    stop = start_spinner("Loading cached corpus")
    try:
        df: pd.DataFrame = load_cached_corpus(args.cache, format=args.cache_format)
    finally:
        stop()

    ensure_csv_header(args.out)
    decided = load_existing_decisions(args.out) if args.resume else set()
    if decided:
        print(f"Resuming: skipping {len(decided)} term(s) already decided.")

    # keep original JSON order; filter decided without sorting
    todo = [t for t in new_terms if t not in decided]

    print("\nControls: [y]es  [n]o  [s]kip  [q]uit\n")
    WINDOW = 12
    K_FETCH = 6  # overfetch, keep first 2 unique

    for i, term in enumerate(todo, 1):
        try:
            windows = query_base_windows(df=df, term=term, k=K_FETCH, window=WINDOW)
        except Exception as e:
            print("=" * 80)
            print(f"[{i}/{len(todo)}] TERM: {term}")
            print(f"(Error obtaining contexts: {e})")
            windows = []

        contexts = extract_context_strings(windows, max_n=2)

        print("=" * 80)
        print(f"[{i}/{len(todo)}] TERM: {term}")
        if contexts:
            for j, c in enumerate(contexts, 1):
                print(f"\nContext {j}:\n  ... {c} ...")
        else:
            print("Contexts: (no occurrences found in cached corpus)")

        while True:
            choice = prompt("\nDecision [y/n/s/q]: ").lower()
            if choice in ("y", "yes"):
                decision = "accept"
            elif choice in ("n", "no"):
                decision = "reject"
            elif choice in ("s", "skip"):
                print("skipped (will reappear next run).")
                break
            elif choice in ("q", "quit"):
                print("Exiting.")
                return
            else:
                print("Enter y/n/s/q")
                continue

            if choice in ("y", "yes", "n", "no"):
                reason = prompt("Reasoning (required): ")
                if not reason:
                    print("Please provide a short reasoning.")
                    continue
                append_row(args.out, {"term": term, "decision": decision, "reasoning": reason, "contexts": contexts})
                print(f"{decision} and saved.")
                break

    print(f"\nDone. Decisions written to: {args.out}")

if __name__ == "__main__":
    main()
