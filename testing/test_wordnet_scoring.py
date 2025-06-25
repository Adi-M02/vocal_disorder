
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import wordnet as wn

try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    wn.ensure_loaded()

# ---------------------------------------------------------------------
# Build weighted “hierarchies” for a single term
# ---------------------------------------------------------------------

_REL_TYPES = ("synonym", "hypernym", "holonym")


def _weighted_nodes_for_synset(
    synset: wn.Synset, rel_type: str
) -> Dict[str, float]:
    """
    Return {lemma_name: weight} for THIS synset, for the requested relation.
    All weights are in (0, 1].
    """
    if rel_type == "synonym":
        # depth = 1  ⇒  weight = (1-0)/1 = 1  (every synonym weight 1)
        return {l.replace("_", " "): 1.0 for l in synset.lemma_names()}

    if rel_type == "hypernym":
        # Breadth-first walk to root
        levels: List[List[wn.Synset]] = [[synset]]
        while levels[-1]:
            next_lvl = []
            for s in levels[-1]:
                next_lvl.extend(s.hypernyms())
            levels.append(next_lvl)

        depth = len(levels)  # includes level 0 (the term itself)
        weights = {}
        for k, lvl in enumerate(levels):
            w = (depth - k) / depth
            for s in lvl:
                for lemma in s.lemma_names():
                    weights[lemma.replace("_", " ")] = max(weights.get(lemma, 0), w)
        return weights

    if rel_type == "holonym":
        # Collect part/substance/member holonyms one step outward,
        # then recurse through THEIR hypernyms to reach the root
        holos = (
            synset.part_holonyms()
            + synset.substance_holonyms()
            + synset.member_holonyms()
        )
        if not holos:
            return {}
        # Build a pseudo-chain: holonym synsets then their hypernyms etc.
        levels: List[List[wn.Synset]] = [holos]
        while levels[-1]:
            next_lvl = []
            for s in levels[-1]:
                next_lvl.extend(s.hypernyms())
            levels.append(next_lvl)

        depth = len(levels)
        weights = {}
        for k, lvl in enumerate(levels):
            w = (depth - k) / depth
            for s in lvl:
                for lemma in s.lemma_names():
                    weights[lemma.replace("_", " ")] = max(weights.get(lemma, 0), w)
        return weights

    raise ValueError(f"Unknown relation type {rel_type}")


def _build_hierarchies(term: str) -> Dict[str, Dict[str, float]]:
    """
    For *all noun senses* of the term, return a dict:
        {rel_type -> {lemma_name -> max_weight}}
    We keep the max weight across senses so that the best match counts.
    """
    hier: Dict[str, Dict[str, float]] = {rt: defaultdict(float) for rt in _REL_TYPES}
    for synset in wn.synsets(term, pos=wn.NOUN):
        for rt in _REL_TYPES:
            nodes = _weighted_nodes_for_synset(synset, rt)
            for lemma, w in nodes.items():
                hier[rt][lemma] = max(hier[rt][lemma], w)
    return hier


# ---------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------


def _best_match_weight(
    hier_q: Dict[str, Dict[str, float]], hier_a: Dict[str, Dict[str, float]]
) -> float:
    """
    Match hierarchies of SAME relation type and return the highest
    product W_q × W_a over *any* overlapping lemma.
    """
    best = 0.0
    for rt in _REL_TYPES:
        q_nodes = hier_q[rt]
        a_nodes = hier_a[rt]
        common = set(q_nodes) & set(a_nodes)
        for lemma in common:
            best = max(best, q_nodes[lemma] * a_nodes[lemma])
    return best


def semantic_similarity(term1: str, term2: str) -> float:
    """
    Linear-decay WordNet similarity between two strings.
    Returns 0-1.
    """
    term1 = term1.lower().replace("_", " ")
    term2 = term2.lower().replace("_", " ")
    if term1 == term2:
        return 1.0

    hier1 = _build_hierarchies(term1)
    hier2 = _build_hierarchies(term2)

    return _best_match_weight(hier1, hier2)


# ---------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python testing/test_wordnet_scoring.py word1 word2")
        sys.exit(1)
    w1, w2 = sys.argv[1:]
    print(f"semantic_similarity({w1!r}, {w2!r}) = {semantic_similarity(w1, w2):.4f}")