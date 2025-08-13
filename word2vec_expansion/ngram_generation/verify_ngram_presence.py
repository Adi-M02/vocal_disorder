#!/usr/bin/env python3
"""
Count 1–3-gram occurrences in a corpus with overlap diagnostics.

- Accepts mixed seed terms: unigrams, bigrams, trigrams (deduped after normalization)
- Overlap-allowed counts: every match is counted
- Non-overlapping counts: leftmost-longest greedy selection (3 > 2 > 1)
- Overlap stats computed two ways:
    (1) *all* matches (unigrams included)
    (2) *excluding unigrams* (clearer view of phrase-vs-phrase overlap)

Usage:
  python count_ngrams.py \
      --ngrams seed_terms.json \
      --out ngram_counts.json \
      --db reddit --collection noburp_all \
      [--limit 0] \
      [--save_examples 3]
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# --- project imports ---
sys.path.append("../vocal_disorder")
from utils.load_json import load_json
from utils.load_and_process_docs import process_all_noburp
from utils.text_pipeline import process_text

def normalize_seed_terms(ngrams_path: str):
    """
    Load seed terms JSON and normalize via process_text.
    Keeps only terms of length 1–3 after normalization.
    Returns:
      gram_str_to_tuple: "a b" -> ("a","b")
      grams_by_len: {1: {('a',)}, 2: {...}, 3: {...}}
    """
    data = load_json(ngrams_path)
    raw_terms: List[str] = data.get("seed_terms", [])
    gram_str_to_tuple: Dict[str, Tuple[str, ...]] = {}
    grams_by_len: Dict[int, set] = defaultdict(set)

    for term in raw_terms:
        toks = process_text(term)
        if 1 <= len(toks) <= 3:
            key = " ".join(toks)
            tup = tuple(toks)
            gram_str_to_tuple[key] = tup  # last wins; dedup by normalized string
            grams_by_len[len(tup)].add(tup)
        # silently drop items that normalize to 0 or >3 tokens

    return gram_str_to_tuple, grams_by_len


def find_all_matches(tokens: Sequence[str], grams_by_len: Dict[int, set]) -> List[Tuple[int, int, str]]:
    """
    Return list of all matches (start, end_exclusive, "a b" or "a b c"), allowing overlaps.
    """
    matches: List[Tuple[int, int, str]] = []
    L = len(tokens)
    lengths = sorted(grams_by_len.keys())  # small->large for "all" is fine
    for i in range(L):
        for n in lengths:
            if i + n <= L:
                tup = tuple(tokens[i:i+n])
                if tup in grams_by_len[n]:
                    matches.append((i, i+n, " ".join(tup)))
    return matches


def greedy_non_overlapping_counts(tokens: Sequence[str], grams_by_len: Dict[int, set]) -> Counter:
    """
    Leftmost-longest greedy: prefer 3-gram over 2-gram over unigram.
    """
    counts = Counter()
    L = len(tokens)
    i = 0
    lens_desc = sorted(grams_by_len.keys(), reverse=True)
    while i < L:
        chosen = None
        chosen_len = 0
        for n in lens_desc:
            if i + n <= L and tuple(tokens[i:i+n]) in grams_by_len[n]:
                chosen = " ".join(tokens[i:i+n])
                chosen_len = n
                break
        if chosen:
            counts[chosen] += 1
            i += chosen_len
        else:
            i += 1
    return counts


def _overlap_stats(matches: List[Tuple[int, int, str]], total_tokens: int):
    """
    Compute overlap flags + token-position overlap from a match list.
    """
    cover = [0] * total_tokens
    for s, e, _ in matches:
        for k in range(s, e):
            cover[k] += 1

    flags = []
    overlapping_match_count = 0
    for s, e, _ in matches:
        overlapped = any(cover[k] > 1 for k in range(s, e))
        flags.append(overlapped)
        if overlapped:
            overlapping_match_count += 1
    total_overlap_positions = sum(1 for c in cover if c > 1)
    return flags, overlapping_match_count, total_overlap_positions


def main():
    ap = argparse.ArgumentParser(description="Count 1–3-gram occurrences with overlap stats.")
    ap.add_argument("--ngrams", required=True, help="Path to JSON with {'seed_terms': [...]} (any mix of 1/2/3-grams).")
    ap.add_argument("--out", required=True, help="Path to write output JSON.")
    ap.add_argument("--db", default="reddit", help="MongoDB database name (default: reddit).")
    ap.add_argument("--collection", default="noburp_all", help="MongoDB collection (default: noburp_all).")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process this many docs for a quick pass.")
    ap.add_argument("--save_examples", type=int, default=3, help="How many overlap example docs to include.")
    args = ap.parse_args()

    gram_str_to_tuple, grams_by_len = normalize_seed_terms(args.ngrams)
    if not grams_by_len:
        raise SystemExit("No 1–3-gram seed terms remained after normalization. Check your seed list and tokenization.")

    # Initialize counters so zero-count keys appear in output
    base = {k: 0 for k in gram_str_to_tuple.keys()}
    overlap_allowed = Counter(base.copy())
    non_overlapping = Counter(base.copy())
    overlapped_instances_all = Counter(base.copy())
    overlapped_instances_excl1 = Counter(base.copy())
    doc_freq = Counter(base.copy())

    docs = process_all_noburp()

    total_docs = 0
    total_tokens = 0

    # overlap summaries
    docs_with_any_overlap_all = 0
    total_overlapping_match_instances_all = 0
    total_overlap_positions_all = 0

    docs_with_any_overlap_excl1 = 0
    total_overlapping_match_instances_excl1 = 0
    total_overlap_positions_excl1 = 0

    overlap_examples_all = []
    overlap_examples_excl1 = []

    for idx, tokens in enumerate(docs):
        if args.limit and idx >= args.limit:
            break
        total_docs += 1
        total_tokens += len(tokens)

        # All matches (1–3 grams)
        matches_all = find_all_matches(tokens, grams_by_len)

        # Counts + doc freq (all)
        if matches_all:
            seen = set()
            for _, _, g in matches_all:
                overlap_allowed[g] += 1
                seen.add(g)
            for g in seen:
                doc_freq[g] += 1

            # Overlap stats: ALL
            flags_all, overlapping_count_all, overlap_pos_all = _overlap_stats(matches_all, len(tokens))
            total_overlapping_match_instances_all += overlapping_count_all
            total_overlap_positions_all += overlap_pos_all
            if overlapping_count_all > 0:
                docs_with_any_overlap_all += 1
                if len(overlap_examples_all) < args.save_examples:
                    example_matches = [{"ngram": g, "span": [s, e]} for (s, e, g), f in zip(matches_all, flags_all) if f]
                    overlap_examples_all.append({
                        "doc_index": idx,
                        "tokens": tokens,
                        "overlapping_matches": example_matches
                    })

            # Per-ngram overlapped instance counts (ALL)
            for (s, e, g), f in zip(matches_all, flags_all):
                if f:
                    overlapped_instances_all[g] += 1

        # Overlap stats: EXCLUDING UNIGRAMS
        if 1 in grams_by_len:
            # filter out matches whose length==1
            matches_excl1 = [(s, e, g) for (s, e, g) in matches_all if (e - s) > 1]
        else:
            matches_excl1 = matches_all  # no unigrams present anyway

        if matches_excl1:
            flags_excl1, overlapping_count_excl1, overlap_pos_excl1 = _overlap_stats(matches_excl1, len(tokens))
            total_overlapping_match_instances_excl1 += overlapping_count_excl1
            total_overlap_positions_excl1 += overlap_pos_excl1
            if overlapping_count_excl1 > 0:
                docs_with_any_overlap_excl1 += 1
                if len(overlap_examples_excl1) < args.save_examples:
                    example_matches = [{"ngram": g, "span": [s, e]} for (s, e, g), f in zip(matches_excl1, flags_excl1) if f]
                    overlap_examples_excl1.append({
                        "doc_index": idx,
                        "tokens": tokens,
                        "overlapping_matches": example_matches
                    })
            # Per-ngram overlapped instance counts (EXCL 1-grams)
            for (s, e, g), f in zip(matches_excl1, flags_excl1):
                if f:
                    overlapped_instances_excl1[g] += 1

        # Non-overlapping (leftmost-longest)
        nonov = greedy_non_overlapping_counts(tokens, grams_by_len)
        for g, c in nonov.items():
            non_overlapping[g] += c

    # Build JSON
    now = datetime.now().isoformat(timespec="seconds")
    out = {
        "meta": {
            "timestamp": now,
            "db": args.db,
            "collection": args.collection,
            "docs_scanned": total_docs,
            "total_tokens": total_tokens,
            "seed_terms_file": str(Path(args.ngrams).resolve()),
            "limit": args.limit,
        },
        "overlap_summary_all": {
            "docs_with_any_overlap": docs_with_any_overlap_all,
            "total_overlapping_match_instances": total_overlapping_match_instances_all,
            "total_overlap_token_positions": total_overlap_positions_all,
            "examples": overlap_examples_all,
        },
        "overlap_summary_excl_unigrams": {
            "docs_with_any_overlap": docs_with_any_overlap_excl1,
            "total_overlapping_match_instances": total_overlapping_match_instances_excl1,
            "total_overlap_token_positions": total_overlap_positions_excl1,
            "examples": overlap_examples_excl1,
        },
        "counts": {}
    }

    for g in sorted(gram_str_to_tuple.keys()):
        tup = gram_str_to_tuple[g]
        n = len(tup)
        total_all = overlap_allowed[g]
        total_nonov = non_overlapping[g]
        out["counts"][g] = {
            "tokens": list(tup),
            "len": n,
            "doc_freq": doc_freq[g],
            "total_overlap_allowed": total_all,
            "total_non_overlapping": total_nonov,
            "overlapped_instances_all": overlapped_instances_all[g],
            "overlap_rate_instances_all": (overlapped_instances_all[g] / total_all) if total_all else 0.0,
            "overlapped_instances_excl_unigrams": overlapped_instances_excl1[g],
            "overlap_rate_instances_excl_unigrams": (overlapped_instances_excl1[g] / total_all) if total_all else 0.0
        }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Docs: {total_docs} | Tokens: {total_tokens}")
    print(f"[ALL] Any-overlap docs: {docs_with_any_overlap_all} | Overlap instances: {total_overlapping_match_instances_all}")
    print(f"[NO 1-grams] Any-overlap docs: {docs_with_any_overlap_excl1} | Overlap instances: {total_overlapping_match_instances_excl1}")
    print("Done.")


if __name__ == "__main__":
    main()
