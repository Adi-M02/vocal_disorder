#!/usr/bin/env python3
"""
Interactive reviewer (cached corpus, NDJSON decisions)
- Input is .ndjson where each line looks like:
  {"seed":"pneumothorax","candidate":"collapse_lung","accepted":true,"decision":"accept",...}
- Keeps the same UI and CSV format as your minimal reviewer:
  Prompts: [y]es/[n]o/[s]kip/[q]uit + required reasoning
  CSV: term,decision,reasoning,context1,context2
- Includes ONLY unique candidates that are ALWAYS rejected across all lines
  (observed decision set is exactly {"reject"}), and excludes ONLY seed terms.
- Preserves the order of first appearance in the .ndjson.
- Shows up to 2 corpus contexts via query_base_windows (window=12, k=6).
"""

import argparse
import csv
import json
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Project imports
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
    """
    query_base_windows returns like: [['...'], ['...']]
    Take first string per doc, dedup, cap at max_n.
    """
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

# ---- NDJSON parsing ----
def normalize_decision(decision_val, accepted_val) -> str | None:
    """
    Normalize decision to "accept" or "reject".
    Priority:
      1) explicit string decision if valid
      2) accepted boolean if present
    """
    if isinstance(decision_val, str):
        d = decision_val.strip().lower()
        if d in {"accept", "accepted"}:
            return "accept"
        if d in {"reject", "rejected"}:
            return "reject"
    if isinstance(accepted_val, bool):
        return "accept" if accepted_val else "reject"
    return None

def load_always_rejected_ordered(ndjson_path: Path) -> Tuple[List[str], Set[str], int, int]:
    """
    Read .ndjson and return:
      - candidates_ordered: unique candidates that are always rejected, ordered by first appearance
      - seeds: set of all seed values seen
      - total_unique_candidates: count of unique candidates in file
      - excluded_because_seed: count of always-rejected candidates that got excluded for being seeds
    """
    if not ndjson_path.exists():
        raise FileNotFoundError(f"Decisions file not found: {ndjson_path}")

    decisions_by_cand: Dict[str, Set[str]] = {}
    first_index: Dict[str, int] = {}
    seeds: Set[str] = set()
    line_idx = 0

    with ndjson_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            line_idx += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                # skip malformed line
                continue

            cand = str(row.get("candidate", "") or "").strip()
            seed = str(row.get("seed", "") or "").strip()
            if seed:
                seeds.add(seed)
            if not cand:
                continue

            norm = normalize_decision(row.get("decision"), row.get("accepted"))
            if norm is None:
                continue

            if cand not in decisions_by_cand:
                decisions_by_cand[cand] = set()
                first_index[cand] = line_idx
            decisions_by_cand[cand].add(norm)

    total_unique_candidates = len(decisions_by_cand)

    # Always rejected = exactly {"reject"}
    always_rejected = [c for c, ds in decisions_by_cand.items() if ds == {"reject"}]

    # Exclude ONLY seed terms; preserve order by first appearance
    always_rejected.sort(key=lambda c: first_index.get(c, 10**12))
    filtered = [c for c in always_rejected if c not in seeds and c != ""]

    excluded_because_seed = sum(1 for c in always_rejected if c in seeds)

    return filtered, seeds, total_unique_candidates, excluded_because_seed

# ------- Main -------
def main():
    ap = argparse.ArgumentParser(description="Interactive manual eval from NDJSON decisions with cached corpus contexts.")
    ap.add_argument("--decisions", default='testing/llm_addition_testing/10-2_full/single_term_eval_corpusctx_10_02_23_38/decisions.ndjson', type=Path, help="Path to .ndjson file of decisions.")
    ap.add_argument("--cache", default="base_ngram_cache_with_details.parquet", type=str, help="Cached corpus path for load_cached_corpus.")
    ap.add_argument("--cache-format", choices=["parquet", "jsonl"], default="parquet")
    ap.add_argument("--out", default="testing/manual_verification/rejection_eval.csv", type=Path, help="Output CSV (append-only).")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip terms already decided.")
    args = ap.parse_args()

    # Load candidates
    candidates, seeds, total_unique, excluded_seed_count = load_always_rejected_ordered(args.decisions)

    print(f"Unique seeds observed: {len(seeds)}")
    print(f"Unique candidates in file: {total_unique}")
    print(f"Always-rejected candidates to review (after excluding seeds): {len(candidates)}")
    if excluded_seed_count:
        print(f"Excluded because candidate equals a seed: {excluded_seed_count}")

    # Load cached corpus
    stop = start_spinner("Loading cached corpus")
    try:
        df: pd.DataFrame = load_cached_corpus(args.cache, format=args.cache_format)
    finally:
        stop()

    ensure_csv_header(args.out)
    decided = load_existing_decisions(args.out) if args.resume else set()
    if decided:
        print(f"Resuming: skipping {len(decided)} term(s) already decided.")

    # keep NDJSON order; filter decided without sorting
    todo = [t for t in candidates if t not in decided]

    print("\nControls: [y]es  [n]o  [s]kip  [q]uit\n")
    WINDOW = 40
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
