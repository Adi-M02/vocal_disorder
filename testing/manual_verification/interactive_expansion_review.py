#!/usr/bin/env python3
"""
Interactive reviewer for expanded terms (CSV-only, Excel-friendly), with seed exclusion.

What it does
------------
- Input: expansions JSON mapping {seed: [expanded_terms...]}
- "New terms" = terms found in any expansion list that are NOT in the exclude set and not empty.
  Exclude set = expansions' seed keys + optional --seed-list + optional --exclude-terms.
- Shows N context snippets for each term (from n-grammed docs) + all seed(s) that produced it.
- Prompts: [a]ccept / [r]eject / [s]kip / [q]uit
- Appends a CSV row for BOTH accepted and rejected items:
    term, seeds, status, reasoning, context1..contextN, decided_at
- --resume skips terms already decided in the CSV.

Seed exclusion
--------------
- By default, this script **always ignores expansion candidates that are also seed keys**.
- You can pass a global/base seed list via --seed-list (JSON array or TXT one-per-line).
- You can also add quick ad-hoc exclusions via --exclude-terms "t1,t2,t3".

Notes
-----
- CSV is UTF-8 with BOM so Excel opens it cleanly.
- Keep --contexts-per-term consistent for one CSV (the script adapts to existing headers).
"""

import argparse
import csv
import json
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---- Project imports (per your environment) ----
sys.path.append("../vocal_disorder")
from utils.load_and_process_docs import process_all_noburp
from utils.text_pipeline import remove_unigram_stopwords
from testing.test_ngram_generation import load_phrasers_from_dir, apply_ngrams


# --------------------------
# Corpus utilities
# --------------------------
def process_ngram_docs(ngram_phraser_dir: str) -> List[List[str]]:
    """Returns tokenized, n-grammed, stopword-removed docs."""
    docs = process_all_noburp(stoplist=False)
    bigram, trigram = load_phrasers_from_dir(ngram_phraser_dir)
    out = []
    for doc in docs:
        doc = apply_ngrams(doc, (bigram, trigram))
        doc = remove_unigram_stopwords(doc)
        out.append(doc)
    return out


# --------------------------
# UI spinner
# --------------------------
def start_spinner(message: str):
    """Simple CLI spinner (non-blocking). Returns a stop() function."""
    stop_flag = {"stop": False}

    def run():
        frames = "|/-\\"
        i = 0
        while not stop_flag["stop"]:
            print(f"\r{message} {frames[i % len(frames)]}", end="", flush=True)
            i += 1
            time.sleep(0.1)
        # clear line
        print("\r" + " " * (len(message) + 2) + "\r", end="", flush=True)

    th = threading.Thread(target=run, daemon=True)
    th.start()

    def stop():
        stop_flag["stop"] = True
        th.join()

    return stop


# --------------------------
# Seed/exclusion helpers
# --------------------------
def load_term_list(path: Path) -> Set[str]:
    """
    Load a set of terms from a JSON array or a UTF-8 text file (one term per line).
    - JSON: ["acid_reflux","bcbs", ...]
    - TXT : each non-empty line is a term
    """
    if not path:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"--seed-list path not found: {path}")

    terms: Set[str] = set()
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for x in obj:
                s = (str(x) if x is not None else "").strip()
                if s:
                    terms.add(s)
        else:
            raise ValueError("--seed-list JSON must be a flat array of strings")
    else:
        # assume text file
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                terms.add(s)
    return terms


# --------------------------
# Expansions handling
# --------------------------
def load_expansions(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists.")
    fixed = {}
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"Value for seed '{k}' must be a list.")
        fixed[k] = [str(x).strip() for x in v]
    return fixed


def compute_candidates(
    expansions: Dict[str, List[str]],
    extra_exclude: Set[str] | None = None,
) -> Tuple[Set[str], Dict[str, Set[str]], Set[str]]:
    """
    Returns:
      - new_terms: unique terms appearing in lists but not in exclude set and not empty.
      - term_to_seeds: map term -> set(seeds) that produced it.
      - excluded_terms: the set of candidate terms that were excluded (e.g., because they are seeds).
    Exclude set = expansions' seed keys ∪ extra_exclude.
    """
    extra_exclude = extra_exclude or set()
    seeds = set(expansions.keys())
    exclude_set = seeds | extra_exclude

    term_to_seeds: Dict[str, Set[str]] = defaultdict(set)
    for seed, terms in expansions.items():
        for t in terms:
            t = (t or "").strip()
            if t:
                term_to_seeds[t].add(seed)

    all_candidates = set(term_to_seeds.keys())
    new_terms = {t for t in all_candidates if t not in exclude_set and t != ""}
    excluded_terms = all_candidates - new_terms
    return new_terms, term_to_seeds, excluded_terms


def build_context_index(
    docs: List[List[str]],
    candidate_terms: Set[str],
    window: int = 12,
    pool_per_term: int = 6,
) -> Dict[str, List[str]]:
    """
    One-pass scan that collects up to `pool_per_term` windows for each candidate term.
    Windows are token spans joined with spaces.
    """
    contexts: Dict[str, List[str]] = {t: [] for t in candidate_terms}
    seen_snips: Dict[str, Set[str]] = {t: set() for t in candidate_terms}
    remaining = set(candidate_terms)

    for doc in docs:
        if not remaining:
            break
        n = len(doc)
        tokens_in_doc = set(doc)
        hits_here = tokens_in_doc & remaining
        if not hits_here:
            continue

        for i, tok in enumerate(doc):
            if tok in remaining:
                lo = max(0, i - window)
                hi = min(n, i + window + 1)
                snippet = " ".join(doc[lo:hi])
                if snippet not in seen_snips[tok]:
                    contexts[tok].append(snippet)
                    seen_snips[tok].add(snippet)
                    if len(contexts[tok]) >= pool_per_term:
                        remaining.discard(tok)
    return contexts


# --------------------------
# CSV utilities
# --------------------------
def ensure_csv_header(csv_path: Path, contexts_per_term: int) -> List[str]:
    """
    Ensure the CSV has a header. If file exists with a header, return it.
    Otherwise create header with the requested number of context columns.
    """
    desired_headers = (
        ["term", "seeds", "status", "reasoning"]
        + [f"context{i}" for i in range(1, contexts_per_term + 1)]
        + ["decided_at"]
    )

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, None)
        if first:
            return first

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(desired_headers)
    return desired_headers


def contexts_count_from_headers(headers: List[str]) -> int:
    return sum(1 for h in headers if h.startswith("context"))


def append_csv_row(csv_path: Path, headers: List[str], record: dict):
    """
    Append a single decision row to CSV, conforming to the existing header.
    """
    k = contexts_count_from_headers(headers)
    row_map = {
        "term": record.get("term", ""),
        "seeds": ";".join(record.get("seeds", [])),
        "status": record.get("status", ""),
        "reasoning": record.get("reasoning") or "",
        "decided_at": record.get("decided_at", ""),
    }
    contexts = record.get("contexts") or []
    for i in range(1, k + 1):
        row_map[f"context{i}"] = contexts[i - 1] if i - 1 < len(contexts) else ""

    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writerow({h: row_map.get(h, "") for h in headers})


def load_existing_decisions_from_csv(csv_path: Path) -> Set[str]:
    """
    Returns terms that already have ANY decision in the CSV (accepted or rejected).
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    decided = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "term" not in reader.fieldnames:
            return set()
        for row in reader:
            t = (row.get("term") or "").strip()
            if t:
                decided.add(t)
    return decided


# --------------------------
# Misc
# --------------------------
def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        return "q"


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Interactive reviewer for expanded terms (CSV-only, seed-excluding).")
    ap.add_argument(
        "--expansions",
        default="testing/llm_addition_testing/8-24/single_with_anchor/single_term_eval_08_24_22_57/accepted_aligned_by_seed.json",
        type=Path,
        help="Path to expansions JSON.",
    )
    ap.add_argument(
        "--ngram-phraser-dir",
        default="testing/ngram_evals/5",
        type=str,
        help="Directory containing bigram/trigram phrasers.",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV path (append-only).",
    )
    ap.add_argument("--contexts-per-term", type=int, default=2, help="Number of snippets to display per term.")
    ap.add_argument("--window", type=int, default=12, help="Token window radius for context.")
    ap.add_argument("--pool-per-term", type=int, default=6, help="Max snippets to pre-collect per term.")
    ap.add_argument("--resume", action="store_true", help="Skip terms already present (any decision) in the CSV.")
    ap.add_argument(
        "--seed-list",
        type=Path,
        default=None,
        help="Optional JSON array or TXT (one per line) of GLOBAL/base seeds to exclude from review.",
    )
    ap.add_argument(
        "--exclude-terms",
        type=str,
        default="",
        help='Optional comma-separated list of extra terms to exclude (e.g., "acid_reflux,bcbs").',
    )
    args = ap.parse_args()

    # Load expansions
    expansions = load_expansions(args.expansions)

    # Build exclude set: expansions' seed keys + optional seed-list + --exclude-terms
    exclude_set: Set[str] = set(expansions.keys())
    if args.seed_list:
        exclude_set |= load_term_list(args.seed_list)
    if args.exclude_terms.strip():
        exclude_set |= {t.strip() for t in args.exclude_terms.split(",") if t.strip()}

    # Compute candidates
    new_terms, term_to_seeds, excluded_terms = compute_candidates(expansions, extra_exclude=exclude_set)

    print(f"Seeds from expansions: {len(expansions)}")
    if args.seed_list:
        print(f"Additional seeds loaded from --seed-list: {len(load_term_list(args.seed_list))}")
    if args.exclude_terms.strip():
        print(f"Ad-hoc excluded terms (--exclude-terms): {len({t.strip() for t in args.exclude_terms.split(',') if t.strip()})}")
    print(f"New unique candidate terms (after exclusion) |T| = {len(new_terms)}")
    print(f"Excluded candidates due to seed/exclude set: {len(excluded_terms)}")

    # Prepare docs (n-grammed)
    stop = start_spinner("Loading & n-gramming documents")
    try:
        docs = process_ngram_docs(args.ngram_phraser_dir)
    finally:
        stop()

    # Build contexts
    stop = start_spinner("Indexing contexts")
    try:
        contexts = build_context_index(
            docs,
            candidate_terms=new_terms,
            window=args.window,
            pool_per_term=args.pool_per_term,
        )
    finally:
        stop()

    # Ensure CSV header; adapt to existing header if present
    headers = ensure_csv_header(args.out, args.contexts_per_term)

    # Resume support
    skip_terms = load_existing_decisions_from_csv(args.out) if args.resume else set()
    if skip_terms:
        print(f"Resuming: will skip {len(skip_terms)} term(s) already decided in {args.out}")

    # Iterate terms in a stable order
    todo_terms = sorted(t for t in new_terms if t not in skip_terms)

    print("\nReview controls: [a]ccept  [r]eject  [s]kip  [q]uit\n")

    for idx, term in enumerate(todo_terms, 1):
        seeds = sorted(term_to_seeds.get(term, []))
        snips = contexts.get(term, [])[: args.contexts_per_term]

        print("=" * 80)
        print(f"[{idx}/{len(todo_terms)}] TERM: {term}")
        print(f"Seeds: {', '.join(seeds) if seeds else '(none found)'}")
        if not snips:
            print("Contexts: (no occurrences found in n-grammed docs)")
        else:
            for i, s in enumerate(snips, 1):
                print(f"\nContext {i}:\n  ... {s} ...")

        while True:
            choice = prompt("\nDecision [a/r/s/q]: ").lower()
            if choice in ("a", "accept"):
                reason = prompt("Reasoning (required, e.g., 'healthcare/diagnostic terminology'): ")
                if not reason:
                    print("Please provide a short reasoning.")
                    continue
                record = {
                    "term": term,
                    "seeds": seeds,
                    "status": "accepted",
                    "reasoning": reason,
                    "contexts": snips,
                    "decided_at": datetime.now().isoformat(timespec="seconds"),
                }
                append_csv_row(args.out, headers, record)
                print("accepted and saved.")
                break

            elif choice in ("r", "reject"):
                reason = prompt("Reasoning (required, e.g., 'off-domain / generic'): ")
                if not reason:
                    print("Please provide a short reasoning.")
                    continue
                record = {
                    "term": term,
                    "seeds": seeds,
                    "status": "rejected",
                    "reasoning": reason or "",
                    "contexts": snips,
                    "decided_at": datetime.now().isoformat(timespec="seconds"),
                }
                append_csv_row(args.out, headers, record)
                print("rejected and saved.")
                break

            elif choice in ("s", "skip"):
                print("skipped (will reappear next run unless decided).")
                break

            elif choice in ("q", "quit"):
                print("Exiting by user request.")
                return

            else:
                print("Please enter a valid option: a/r/s/q")

    print(f"\nDone. Decisions written to: {args.out}")


if __name__ == "__main__":
    main()
