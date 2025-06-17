"""
Analyse an RCPD vocabulary across Reddit documents
--------------------------------------------------

Outputs
=======
* top_terms_per_category.csv  – frequency table of the N most common terms in each category
* cooccur_within_category.csv – term-pair counts + NPMI within the same category
* cooccur_cross_category.csv  – term-pair counts + NPMI across different categories
* tfidf_category_scores.csv   – averaged TF-IDF weight per term within its category

The script relies on project-local helpers already in your repository:
    • return_documents  – wrapper around MongoDB (query_mongo.py)
    • clean_and_tokenize – basic tokeniser (tokenizer.py)
    • spellcheck_token_list – spell-checking wrapper (spellchecker_folder)

Multi-word terms are matched using a simple longest-match scan.  If you have
FlashText or spaCy installed you can swap the matcher implementation.

Usage
-----
python vocab_category_analysis.py \
    --vocab_path vocabulary/rcpd_vocab.json \
    --db reddit --collection noburp_all \
    --subreddits noburp \
    --lemma_lookup resources/lemma_lookup.json \
    --top_n 30
"""

import argparse, json, math, itertools, collections, pathlib, csv, os
from typing import List, Dict, Tuple, Iterable, Counter
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# --- project helpers ----------------------------------------------------
import sys
sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
try:
    from spellchecker_folder.spellchecker import spellcheck_token_list
except ImportError:
    def spellcheck_token_list(tok_list):
        return tok_list  # graceful fallback

# ------------------------------------------------------------------------
# helper: longest-match term extractor (multi-word aware)
# ------------------------------------------------------------------------

def build_term_index(terms: Iterable[str]):
    """Return dict: first_token → list[ (token_len, term) ] sorted by len desc."""
    idx: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)
    for term in terms:
        toks = term.split()
        idx[toks[0]].append((len(toks), term))
    for tok in idx:
        idx[tok].sort(reverse=True)      # longest first
    return idx


def extract_terms(tokens: List[str], term_idx) -> List[str]:
    """Return *all* (possibly overlapping) vocabulary terms found in the token list."""
    found = []
    n = len(tokens)
    i = 0
    while i < n:
        first = tokens[i]
        if first not in term_idx:
            i += 1
            continue
        matched = False
        for L, term in term_idx[first]:
            if i + L <= n and tokens[i:i+L] == term.split():
                found.append(term)
                i += L            # **non-overlapping** longest-match; comment out for overlaps
                matched = True
                break
        if not matched:
            i += 1
    return found

# ------------------------------------------------------------------------
# TF-IDF helper (sklearn)
# ------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_matrix(docs_tokenised: List[List[str]], vocabulary: List[str]):
    """Return (tfidf_matrix, feature_names) where docs are pre-tokenised lists of vocab terms."""
    docs_joined = [" ".join(tok_list) for tok_list in docs_tokenised]
    vect = TfidfVectorizer(vocabulary=vocabulary, tokenizer=str.split, preprocessor=lambda x: x)
    X = vect.fit_transform(docs_joined)
    return X, vect.get_feature_names_out()

# ------------------------------------------------------------------------
# main analysis
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse vocabulary frequencies, co‑occurrence, and TF‑IDF scores")
    parser.add_argument('--vocab_path', default="word2vec_expansion/word2vec_06_13_12_28/grid_0613_1635/word2vec_cbow_cos60_f1/expansions.json")
    parser.add_argument('--db', default="reddit")
    parser.add_argument('--collection', default="noburp_all")
    parser.add_argument('--subreddits', default=["noburp"])
    parser.add_argument('--lemma_lookup', default="testing/lemma_lookup.json")
    parser.add_argument('--top_n', type=int, default=30,
                        help='Top N terms per category (frequency)')
    parser.add_argument('--min_cooc', type=int, default=10,
                        help='Ignore term pairs with <k co‑occurrences')
    args = parser.parse_args()

    # <<< 1) CREATE output folder with mm_dd_HH_MM timestamp
    timestamp = datetime.now().strftime("%m_%d_%H_%M")       # mm_dd_hh_mm
    results_dir = pathlib.Path(__file__).parent / f"results_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    # >>>

    # ---------------- load resources ------------------------------------
    vocab_by_cat: Dict[str, List[str]] = json.load(open(args.vocab_path))
    term2cat = {term.lower(): cat for cat, terms in vocab_by_cat.items() for term in terms}
    all_terms = set(term2cat.keys())
    term_index = build_term_index(all_terms)

    lemma_map = json.load(open(args.lemma_lookup)) if os.path.exists(args.lemma_lookup) else {}

    # ---------------- tokenisation wrapper ------------------------------
    def tok_fn(text: str) -> List[str]:
        base = spellcheck_token_list(clean_and_tokenize(text))
        return [lemma_map.get(t, t) for t in base]

    # ---------------- fetch documents -----------------------------------
    docs_cursor = return_documents(
        db_name=args.db,
        collection_name=args.collection,
        filter_subreddits=args.subreddits,
    )

    doc_tokens_all  = []        # list[list[str]] for tf-idf
    term_counts     = collections.Counter()
    cat_counts      = collections.Counter()
    pair_counts_in  = collections.Counter()   # within-category pairs
    pair_counts_out = collections.Counter()   # cross-category pairs
    doc_freq        = collections.Counter()   # term → #docs containing term
    TOTAL_DOCS      = 0

    for doc in tqdm(docs_cursor, desc="Scanning docs"):
        TOTAL_DOCS += 1
        tokens = tok_fn(doc)
        doc_terms = extract_terms(tokens, term_index)
        if not doc_terms:
            continue
        doc_tokens_all.append(doc_terms)

        term_counts.update(doc_terms)
        cat_counts.update(term2cat[t] for t in doc_terms)
        doc_freq.update(set(doc_terms))

        # co-occurrence (unique pairs per doc)
        uniq_terms = sorted(set(doc_terms))
        for a, b in itertools.combinations(uniq_terms, 2):
            if term2cat[a] == term2cat[b]:
                pair_counts_in[(a, b)] += 1
            else:
                pair_counts_out[(a, b)] += 1

    # ---------------- export top terms per category ---------------------
    top_rows = []
    for cat, terms in vocab_by_cat.items():
        subcnt = {t: term_counts[t] for t in terms}
        subtotal = sum(subcnt.values()) or 1
        for term, cnt in collections.Counter(subcnt).most_common(args.top_n):
            top_rows.append({
                'category': cat,
                'term': term,
                'count': cnt,
                'share_within_category': cnt / subtotal
            })
    pd.DataFrame(top_rows).to_csv(results_dir / 'top_terms_per_category.csv', index=False)  # <<<

    # ---------------- helper: NPMI --------------------------------------
    def npmi(a, b, n_ab):
        p_ab = n_ab / TOTAL_DOCS
        if p_ab == 0:
            return 0.0
        p_a = doc_freq[a] / TOTAL_DOCS
        p_b = doc_freq[b] / TOTAL_DOCS
        pmi = math.log(p_ab / (p_a * p_b) + 1e-12, 2)
        return pmi / (-math.log(p_ab, 2))

    def export_pairs(counter: Counter, fname: str):
        rows = []
        for (a, b), cnt in counter.items():
            if cnt < args.min_cooc:
                continue
            rows.append({
                'term_a': a,
                'term_b': b,
                'count': cnt,
                'npmi': npmi(a, b, cnt)
            })
        # --- new: sort by npmi descending ---
        rows.sort(key=lambda r: r['npmi'], reverse=True)
        pd.DataFrame(rows).to_csv(results_dir / fname, index=False)

    export_pairs(pair_counts_in,  'cooccur_within_category.csv')
    export_pairs(pair_counts_out, 'cooccur_cross_category.csv')

    # ---------------- TF-IDF by category --------------------------------
    tfidf_matrix, feat_names = build_tfidf_matrix(doc_tokens_all, list(all_terms))
    import numpy as np
    avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

    tf_rows = []
    for term, w in zip(feat_names, avg_weights):
        tf_rows.append({'term': term,
                        'category': term2cat[term],
                        'avg_tfidf': float(w)})
    (pd.DataFrame(tf_rows)
       .sort_values(['category', 'avg_tfidf'], ascending=[True, False])
       .to_csv(results_dir / 'tfidf_category_scores.csv', index=False))  # <<<

    print(f"\n✓ Analysis complete; results in folder:\n  {results_dir}")

if __name__ == "__main__":
    main()
