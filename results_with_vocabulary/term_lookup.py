#!/usr/bin/env python3
"""
term_cooccur_lookup.py

Given a target vocabulary term, compute its top co-occurring vocabulary terms
in the RCPD Reddit corpus, ranked by NPMI, and display their categories.

Usage:
  python term_cooccur_lookup.py \
    --vocab_path path/to/expansions.json \
    --db reddit --collection noburp_all --subreddits noburp \
    --term "botox" --top_k 20 --min_cooc 5
"""
import argparse
import json
import math
from collections import Counter, defaultdict
import sys

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# project imports
sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# Ensure stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


def build_term_index(terms):
    """Build longest-match index for extractor."""
    idx = defaultdict(list)
    for term in terms:
        toks = term.split()
        idx[toks[0]].append((len(toks), term))
    for k in idx:
        idx[k].sort(reverse=True)
    return idx


def extract_terms(tokens, term_idx):
    """Longest-match extract vocabulary terms from token list."""
    found = []
    i = 0
    n = len(tokens)
    while i < n:
        first = tokens[i]
        if first not in term_idx:
            i += 1
            continue
        matched = False
        for L, term in term_idx[first]:
            if i + L <= n and tokens[i:i+L] == term.split():
                found.append(term)
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return found


def compute_npmi(count_xy, count_x, count_y, total_docs):
    """Compute normalized pointwise mutual information."""
    p_xy = count_xy / total_docs
    if p_xy == 0:
        return 0.0
    p_x = count_x / total_docs
    p_y = count_y / total_docs
    pmi = math.log(p_xy / (p_x * p_y) + 1e-12)
    return pmi / (-math.log(p_xy + 1e-12))


def main():
    parser = argparse.ArgumentParser(description="Lookup top co-occurring terms by NPMI")
    parser.add_argument('--vocab_path', required=True,
                        help='JSON of category -> terms')
    parser.add_argument('--db', default='reddit')
    parser.add_argument('--collection', default='noburp_all')
    parser.add_argument('--subreddits', nargs='+', default=['noburp'])
    parser.add_argument('--term', required=True, help='Target vocabulary term')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top co-occurring terms to show')
    parser.add_argument('--min_cooc', type=int, default=5,
                        help='Minimum co-occurrence count to include')
    args = parser.parse_args()

    # Load vocabulary and categories
    vocab_by_cat = json.load(open(args.vocab_path, 'r', encoding='utf-8'))
    term2cat = {t: cat for cat, terms in vocab_by_cat.items() for t in terms}
    all_terms = set(term2cat.keys())
    term_idx = build_term_index(all_terms)

    target = args.term
    if target not in all_terms:
        print(f"Error: '{target}' not found in vocabulary.")
        return

    # Iterate documents
    docs = return_documents(db_name=args.db,
                             collection_name=args.collection,
                             filter_subreddits=args.subreddits)
    total_docs = 0
    doc_freq = Counter()
    cooc_counts = Counter()

    for doc in tqdm(docs, desc='Scanning docs'):
        text = str(doc)
        tokens = clean_and_tokenize(text)
        if not tokens:
            continue
        total_docs += 1
        terms = extract_terms(tokens, term_idx)
        if target not in terms:
            # skip docs without the target
            for t in terms:
                doc_freq[t] += 1
            continue
        # count doc-level term frequency
        uniq_terms = set(terms)
        for t in uniq_terms:
            doc_freq[t] += 1
        # count co-occurrences with target
        for t in uniq_terms:
            if t != target:
                cooc_counts[t] += 1

    # Compute NPMI per co-occurring term
    results = []
    count_x = doc_freq[target]
    for term, count_xy in cooc_counts.items():
        if count_xy < args.min_cooc:
            continue
        count_y = doc_freq[term]
        npmi = compute_npmi(count_xy, count_x, count_y, total_docs)
        results.append((term, term2cat.get(term, 'UNKNOWN'), count_xy, npmi))

    # Sort by NPMI descending
    results.sort(key=lambda x: x[3], reverse=True)

    print(f"Top {args.top_k} co-occurring terms with '{target}' (min_cooc={args.min_cooc}):")
    print(f"{'Term':<20} {'Category':<20} {'Cooc':<5} {'NPMI':>6}")
    print('-'*55)
    for term, cat, count_xy, npmi in results[:args.top_k]:
        print(f"{term:<20} {cat:<20} {count_xy:<5} {npmi:6.3f}")

if __name__ == '__main__':
    main()
