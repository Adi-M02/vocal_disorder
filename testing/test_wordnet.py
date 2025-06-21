# similarity_rank.py
# -----------------------------------------------------------
# Rank a list of terms by semantic similarity to a query term
# using ❶ TF-IDF weighting on a background corpus and
# ❷ WordNet lexical–hierarchy overlap (synonym / hypernym /
#    holonym).  The final score is:
#
#   score(term) = TFIDF(term) × max_{i,j} W_Qi · W_Tj
#
# where W_k=(depth−k)/depth for the k-th node in a term’s tree.
# -----------------------------------------------------------

from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple
import math, re
import sys

import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append('../vocal_disorder')
from query_mongo import return_documents

# ─── 0. one-time setup ──────────────────────────────────────
#   $ python -m nltk.downloader wordnet stopwords
#   (make sure to run once)

# ─── 1. TF-IDF model on any background corpus ──────────────
def build_vectorizer(docs: List[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        lowercase=True, token_pattern=r"[A-Za-z]\w{2,}",
        stop_words="english",  # remove a few high-freq words
        sublinear_tf=True      # log-TF
    )
    vec.fit(docs)
    return vec

# ─── 2. WordNet helpers ─────────────────────────────────────
def _best_synset(term: str, pos="n"):
    """Pick *one* synset (first sense) – simple but effective."""
    synsets = wn.synsets(term, pos=pos)
    return synsets[0] if synsets else None

def hierarchy(term: str, pos="n") -> List[Tuple[str, float]]:
    """
    Build <lemma, weight> list combining synonyms + all
    hypernyms + all holonyms (same POS).
    W_k = (depth - k) / depth
    """
    s = _best_synset(term, pos)
    if s is None:
        return []

    # Step 1: linearised tree of lemmas ----------
    chain: List[str] = []
    seen = set()

    #   1a. synonym lemmas (depth k=0)
    for l in s.lemmas():
        if l.name() not in seen:
            chain.append(l.name().lower())
            seen.add(l.name())

    #   1b. walk up hypernym chain
    def walk(syn):
        for hyp in syn.hypernyms():
            for l in hyp.lemmas():
                if l.name() not in seen:
                    chain.append(l.name().lower()); seen.add(l.name())
            walk(hyp)
    walk(s)

    #   1c. holonyms (part-of / member-of)
    for rel in (s.member_holonyms() + s.part_holonyms()):
        for l in rel.lemmas():
            if l.name() not in seen:
                chain.append(l.name().lower()); seen.add(l.name())

    # Step 2: assign geometric weights ----------
    depth = len(chain) - 1  # max k
    if depth <= 0:
        return [(lemma, 1.0) for lemma in chain]

    w_list = [(lemma, (depth - k) / depth) for k, lemma in enumerate(chain)]
    return w_list

def lexical_overlap(query_h: List[Tuple[str,float]],
                    term_h:  List[Tuple[str,float]]) -> float:
    """
    Maximum weight product for any exact-lemma match
    between the two hierarchies (same POS assumed).
    """
    q_map = {l:w for l,w in query_h}
    best = 0.0
    for l,w_t in term_h:
        if l in q_map:
            best = max(best, w_t * q_map[l])
    return best     # 0 if no overlap

# ─── 3. Main ranking routine ────────────────────────────────
def rank_terms(query: str,
               candidates: List[str],
               tfidf_vec: TfidfVectorizer,
               pos="n") -> List[Tuple[str,float]]:
    """
    Return (term, score) sorted descending.
    """
    # build hierarchies once
    h_Q = hierarchy(query, pos=pos)
    if not h_Q:
        raise ValueError(f"No WordNet entry for query term ‘{query}’")

    # TF-IDF look-up convenience
    idf = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_))
    avg_idf = sum(idf.values()) / len(idf)

    ranked = []
    for term in candidates:
        h_T = hierarchy(term, pos=pos)
        if not h_T:
            continue

        # TFIDF(term) = TF (1) × IDF
        tfidf = idf.get(term.lower(), avg_idf)   # unseen → mean-IDF
        overlap = lexical_overlap(h_Q, h_T)
        score = tfidf * overlap
        ranked.append((term, score))

    # highest score first
    return sorted(ranked, key=lambda x: x[1], reverse=True)

# ─── 4. Demo / usage ────────────────────────────────────────
if __name__ == "__main__":
    # quick dummy corpus – replace with *your* background docs
    corpus = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"]
    )
    vec = build_vectorizer(corpus)

    query = "bloat"
    terms = ["gi issues", "gastroesophageal reflux disease", "heartburn", "reflux", "r", "pressure"]
    print("Ranking vs query:", query)
    for t, s in rank_terms(query, terms, vec):
        print(f"{t:>10s} : {s:.4f}")
