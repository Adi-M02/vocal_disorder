#!/usr/bin/env python3
"""
Promote semantically-related expansion terms into the global seed list using WordNet.

Usage:
  python augment_seeds_with_wordnet.py \
      --expansions path/to/expansions.json \
      --outdir path/to/output_dir \
      [--hyper_depth 1] [--hypo_depth 1] [--hol_depth 1] \
      [--cohypo_up 1] [--cohypo_down 2] \
      [--adj_hops 2] [--closure_iters 1] \
      [--min_ancestor_depth 0] [--pos_gate_nouns 1] \
      [--max_per_seed 0] [--sort]

Inputs:
- expansions.json : { "<seed>": ["term1","term2",...], ... }

Outputs (written under --outdir/added_terms_to_eval/):
- new_eval_set.json : { "seed_terms": [...] }  (deduped, optionally sorted)
- wordnet_addition_log : text log of ALL additions per seed (one line per seed)
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet

# -----------------------------
# Normalization helpers
# -----------------------------
def norm(tok: str) -> str:
    return tok.strip().lower()

def to_wordnet(tok: str) -> str:
    return norm(tok).replace(" ", "_")

# -----------------------------
# WordNet graph utilities
# -----------------------------
def _lemma_names(synsets: Iterable[wordnet.synset]) -> Set[str]:
    out = set()
    for ss in synsets:
        for l in ss.lemmas():
            out.add(l.name().lower())
    return out

def _derivational_from_synsets(synsets: Iterable[wordnet.synset]) -> Set[str]:
    out = set()
    for ss in synsets:
        for l in ss.lemmas():
            for d in l.derivationally_related_forms():
                out.add(d.name().lower())
    return out

def _bfs(start: Set[wordnet.synset], step_fn, depth: int) -> Set[wordnet.synset]:
    """BFS over synset graph up to 'depth' edges; returns all visited synsets (incl. start)."""
    if depth <= 0 or not start:
        return set(start)
    seen = set(start)
    frontier = set(start)
    for _ in range(depth):
        nxt = set()
        for ss in frontier:
            for nb in step_fn(ss):
                if nb not in seen:
                    seen.add(nb); nxt.add(nb)
        frontier = nxt
    return seen

def synsets_any(word: str) -> List[wordnet.synset]:
    return wordnet.synsets(to_wordnet(word))  # all POS

def synsets_pos(word: str, pos: str) -> List[wordnet.synset]:
    return wordnet.synsets(to_wordnet(word), pos=pos)

def synonyms(word: str) -> Set[str]:
    return _lemma_names(synsets_any(word))

def derivationals(word: str) -> Set[str]:
    return _derivational_from_synsets(synsets_any(word))

def adjective_neighbors(seeds: Iterable[str], hops: int = 2) -> Set[str]:
    """
    Build an adjective neighborhood by walking 'similar_tos' and 'also_sees'
    from ADJ synsets of (seed, seed+'ed', seed+'ing', and each seed's synonyms).
    """
    if hops <= 0:
        return set()
    start_synsets = set()
    for s in seeds:
        forms = {s, f"{s}ed", f"{s}ing"} | synonyms(s)
        for f in forms:
            for ss in synsets_pos(f, wordnet.ADJ):
                start_synsets.add(ss)
    def step(ss):
        return list(ss.similar_tos()) + list(ss.also_sees())
    reached = _bfs(start_synsets, step, depth=hops)
    return _lemma_names(reached)

# -----------------------------
# Ancestor depth-gated relations
# -----------------------------
def hypernyms(word: str, depth: int = 1, *, min_ancestor_depth: int = 0) -> Set[str]:
    """N/V hypernyms up to depth hops. Filters ancestors by WordNet min_depth >= min_ancestor_depth."""
    ssets = [ss for ss in synsets_any(word) if ss.pos() in ("n", "v")]
    if not ssets or depth <= 0:
        return set()
    def step(ss):
        out = list(ss.hypernyms())
        if ss.pos() == "n":
            out += list(ss.instance_hypernyms())
        return out
    reached = _bfs(set(ssets), step, depth)
    if min_ancestor_depth > 0:
        reached = {ss for ss in reached if hasattr(ss, "min_depth") and ss.min_depth() >= min_ancestor_depth}
    return _lemma_names(reached)

def hyponyms(word: str, depth: int = 1) -> Set[str]:
    """N/V hyponyms up to depth hops."""
    ssets = [ss for ss in synsets_any(word) if ss.pos() in ("n", "v")]
    if not ssets or depth <= 0:
        return set()
    def step(ss):
        return list(ss.hyponyms())
    reached = _bfs(set(ssets), step, depth)
    return _lemma_names(reached)

def holonyms(word: str, depth: int = 1) -> Set[str]:
    """Holonyms (member/part/substance) for NOUN senses up to depth hops."""
    ssets = synsets_pos(word, wordnet.NOUN)
    if not ssets or depth <= 0:
        return set()
    def step(ss):
        return list(ss.member_holonyms()) + list(ss.part_holonyms()) + list(ss.substance_holonyms())
    reached = _bfs(set(ssets), step, depth)
    return _lemma_names(reached)

def cohyponyms(word: str, up: int = 1, down: int = 1, *, noun_only: bool = True,
               min_ancestor_depth: int = 0) -> Set[str]:
    """
    Siblings via hypernym(s): go UP 'up' hypernym hops from the seed's synsets,
    then DOWN 'down' hyponym hops; ancestors are filtered by min_depth if provided.
    Excludes the seed's own lemmas (keeps true siblings).
    """
    ssets = [ss for ss in synsets_any(word) if (ss.pos()=="n" if noun_only else ss.pos() in ("n","v"))]
    if not ssets or up <= 0 or down <= 0:
        return set()
    def up_step(ss):
        out = list(ss.hypernyms())
        if ss.pos() == "n":
            out += list(ss.instance_hypernyms())
        return out
    ancestors = _bfs(set(ssets), up_step, up)
    if min_ancestor_depth > 0:
        ancestors = {a for a in ancestors if hasattr(a, "min_depth") and a.min_depth() >= min_ancestor_depth}
    def down_step(ss):
        return list(ss.hyponyms())
    desc = set()
    for a in ancestors:
        desc |= _bfs({a}, down_step, down)
    names = _lemma_names(desc)
    seed_lemmas = _lemma_names(ssets)
    return {w for w in names if w not in seed_lemmas}

# -----------------------------
# Neighborhood composer
# -----------------------------
def expand_wordnet_neighborhood(
    anchors: Iterable[str],
    *,
    include_syn=True,
    include_deriv=True,
    hyper_depth=1,
    hypo_depth=1,
    hol_depth=1,
    cohypo_up=0,
    cohypo_down=0,
    adj_hops=2,
    min_ancestor_depth=0,
) -> Set[str]:
    """Build a set of WordNet-related lemmas around the given anchor words."""
    anchors = {norm(a) for a in anchors}
    out: Set[str] = set()

    if include_syn:
        for a in anchors:
            out |= synonyms(a)
    if include_deriv:
        for a in anchors:
            out |= derivationals(a)
    if hyper_depth > 0:
        for a in anchors:
            out |= hypernyms(a, hyper_depth, min_ancestor_depth=min_ancestor_depth)
    if hypo_depth > 0:
        for a in anchors:
            out |= hyponyms(a, hypo_depth)
    if hol_depth > 0:
        for a in anchors:
            out |= holonyms(a, hol_depth)
    if cohypo_up > 0 and cohypo_down > 0:
        for a in anchors:
            out |= cohyponyms(a, up=cohypo_up, down=cohypo_down,
                              noun_only=True, min_ancestor_depth=min_ancestor_depth)
    if adj_hops > 0:
        out |= adjective_neighbors(anchors, hops=adj_hops)

    return {norm(x) for x in out}

# -----------------------------
# POS helpers (noun gating)
# -----------------------------
def dominant_pos_is_noun(word: str) -> bool:
    """Return True if the seed's dominant POS (by synset count) is noun."""
    ss = synsets_any(word)
    if not ss:
        return False
    from collections import Counter
    cnt = Counter(s.pos() for s in ss)
    noun_count = cnt.get('n', 0)
    other = cnt.get('v', 0) + cnt.get('a', 0) + cnt.get('s', 0) + cnt.get('r', 0)
    return noun_count >= other

def has_noun_sense(word: str) -> bool:
    """Does the candidate have at least one noun synset?"""
    return any(ss.pos() == 'n' for ss in synsets_any(word))

# -----------------------------
# Selection logic per seed
# -----------------------------
def select_candidates_for_seed(
    seed: str,
    expansion_terms: List[str],
    *,
    hyper_depth: int = 1,
    hypo_depth: int = 1,
    hol_depth: int = 1,
    cohypo_up: int = 0,
    cohypo_down: int = 0,
    adj_hops: int = 2,
    closure_iters: int = 1,
    min_ancestor_depth: int = 0,
    pos_gate_nouns: bool = True,
    max_per_seed: int = 0  # 0 = unlimited
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Return (accepted_terms, reasons_map) for a single seed.
    - Optional POS gating: if the seed's dominant POS is noun, only noun candidates are considered.
    """
    seed_n = norm(seed)
    cands = [norm(t) for t in expansion_terms]

    # --- POS gate for nouns (optional) ---
    if pos_gate_nouns and dominant_pos_is_noun(seed_n):
        cands = [t for t in cands if has_noun_sense(t)]

    cand_set = set(cands)
    accepted: Set[str] = set()
    reasons: Dict[str, str] = {}

    # 1) Initial neighborhood around the seed
    seed_neigh = expand_wordnet_neighborhood(
        [seed_n],
        include_syn=True,
        include_deriv=True,
        hyper_depth=hyper_depth,
        hypo_depth=hypo_depth,
        hol_depth=hol_depth,
        cohypo_up=cohypo_up,
        cohypo_down=cohypo_down,
        adj_hops=adj_hops,
        min_ancestor_depth=min_ancestor_depth,
    )

    for t in cands:
        if t in seed_neigh:
            accepted.add(t)
            reasons[t] = "seed-wordnet"

    # 2) Tight semantic closure from admitted pivots (clamped to 1-hop neighborhoods)
    if closure_iters > 0 and accepted:
        frontier = deque(sorted(accepted))
        seen_frontier = set(frontier)
        steps = 0
        while frontier and steps < closure_iters:
            size_this_round = len(frontier)
            for _ in range(size_this_round):
                pivot = frontier.popleft()
                pivot_neigh = expand_wordnet_neighborhood(
                    [pivot],
                    include_syn=True,
                    include_deriv=True,
                    hyper_depth=max(0, min(1, hyper_depth)),
                    hypo_depth=max(0, min(1, hypo_depth)),
                    hol_depth=max(0, min(1, hol_depth)),
                    cohypo_up=max(0, min(1, cohypo_up)),
                    cohypo_down=max(0, min(1, cohypo_down)),
                    adj_hops=max(0, min(1, adj_hops)),
                    min_ancestor_depth=min_ancestor_depth,
                )
                for t in (cand_set & pivot_neigh) - accepted:
                    accepted.add(t)
                    reasons[t] = f"closure@{steps+1}:{pivot}"
                    if t not in seen_frontier:
                        frontier.append(t)
                        seen_frontier.add(t)
            steps += 1

    if max_per_seed > 0 and len(accepted) > max_per_seed:
        accepted = set(sorted(list(accepted))[:max_per_seed])

    return accepted, reasons

# -----------------------------
# I/O
# -----------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -----------------------------
# Logging helpers
# -----------------------------
def setup_file_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("augment_wn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File-only logger (no console)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    return logger

def log_per_seed_lines(logger: logging.Logger, added_by_seed: Dict[str, List[Tuple[str, str]]]):
    if not added_by_seed:
        logger.info("No additions recorded.")
        return
    maxw = max(len(s) for s in added_by_seed.keys())
    maxw = max(12, min(maxw, 28))  # clamp for readability
    for seed in sorted(added_by_seed.keys()):
        pairs = added_by_seed[seed]
        if not pairs:
            continue
        right = ", ".join(f"{t}({r})" for t, r in sorted(pairs, key=lambda x: (x[1], x[0])))
        logger.info(f"{seed:<{maxw}s} -> {right}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Augment seed terms using WordNet relations over expansions.")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", required=True, help="Directory where outputs are written (a subdir will be created)")
    ap.add_argument("--hyper_depth", type=int, default=4, help="Hypernym hops for seed")
    ap.add_argument("--hypo_depth",  type=int, default=4, help="Hyponym hops for seed")
    ap.add_argument("--hol_depth",   type=int, default=4, help="Holonym hops for seed nouns")
    ap.add_argument("--cohypo_up",   type=int, default=1, help="Co-hyponyms: up hypernym hops (0=off)")
    ap.add_argument("--cohypo_down", type=int, default=4, help="Co-hyponyms: down hyponym hops from each ancestor")
    ap.add_argument("--adj_hops",    type=int, default=4, help="Adjective graph hops")
    ap.add_argument("--closure_iters", type=int, default=4, help="Semantic closure iterations")
    ap.add_argument("--min_ancestor_depth", type=int, default=0,
                    help="Minimum WordNet depth required for ancestor synsets used in hyper/co-hypo (0=off; try 4–6)")
    ap.add_argument("--pos_gate_nouns", type=int, default=1,
                    help="(0=off), when on, if seed's dominant POS is noun, require candidates to have a noun sense")
    ap.add_argument("--max_per_seed", type=int, default=0, help="Optional cap of admitted terms per seed (0=off)")
    ap.add_argument("--sort", action="store_true", help="Sort final seed list alphabetically")
    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # Prepare output directory structure with timestamp
    outdir = Path(args.outdir).expanduser().resolve()
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    eval_dir = outdir / f"added_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_out_path = eval_dir / "new_eval_set.json"
    log_path = eval_dir / "wordnet_addition.log"

    logger = setup_file_logger(log_path)

    base_seeds: List[str] = sorted(norm(s) for s in expansions.keys())
    augmented: Set[str] = set(base_seeds)
    added_by_seed: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    accepted_by_seed = defaultdict(list)
    for seed, cand_list in expansions.items():
        if not isinstance(cand_list, list):
            continue
        accepted, reasons = select_candidates_for_seed(
            seed,
            cand_list,
            hyper_depth=args.hyper_depth,
            hypo_depth=args.hypo_depth,
            hol_depth=args.hol_depth,
            cohypo_up=args.cohypo_up,
            cohypo_down=args.cohypo_down,
            adj_hops=args.adj_hops,
            closure_iters=args.closure_iters,
            min_ancestor_depth=args.min_ancestor_depth,
            pos_gate_nouns=bool(args.pos_gate_nouns),
            max_per_seed=args.max_per_seed,
        )
        for t in sorted(accepted):
            accepted_by_seed[norm(seed)].append((t, reasons.get(t, "accepted")))
            if t not in augmented:
                augmented.add(t)
                added_by_seed[norm(seed)].append((t, reasons.get(t, "accepted")))

    out_list = sorted(augmented | set(base_seeds)) if args.sort else list(augmented | set(base_seeds))
    save_json(json_out_path, {"seed_terms": out_list})

    # ---- Write the log (full list, no truncation; no console output) ----
    logger.info("# Additions written by augment_seeds_with_wordnet")
    logger.info(f"# Expansions: {Path(args.expansions).resolve()}")
    logger.info(f"# Output dir: {eval_dir.resolve()}")
    logger.info(f"# JSON out  : {json_out_path.resolve()}")
    logger.info(f"# HyperDepth={args.hyper_depth} HypoDepth={args.hypo_depth} "
                f"HolDepth={args.hol_depth} CoHypoUp={args.cohypo_up} CoHypoDown={args.cohypo_down} "
                f"AdjHops={args.adj_hops} ClosureIters={args.closure_iters} "
                f"MinAncestorDepth={args.min_ancestor_depth} PosGateNouns={bool(args.pos_gate_nouns)} "
                f"MaxPerSeed={args.max_per_seed} Sort={bool(args.sort)}")
    logger.info("## new to eval set")
    log_per_seed_lines(logger, added_by_seed)
    logger.info("")
    logger.info("## all accepted by wordnet, including already in eval set")
    log_per_seed_lines(logger, accepted_by_seed)
    logger.info("")
    total_added = sum(len(v) for v in added_by_seed.values())
    logger.info(f"# Original seeds (from expansion keys): {len(base_seeds)}")
    logger.info(f"# After augmentation (unique terms in eval set): {len(out_list)}")
    logger.info(f"# Total added from expansions via WordNet filters: {total_added}")

if __name__ == "__main__":
    main()
