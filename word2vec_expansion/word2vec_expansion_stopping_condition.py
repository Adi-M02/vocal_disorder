import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
import itertools
import numpy as np
from gensim.models import Word2Vec
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from query_mongo import return_documents
from spellchecker_folder.spellchecker import spellcheck_token_list
from evaluate_expansions_lemmatized import evaluate_terms_performance, load_user_list, parse_ngram_filter
from nltk.corpus import stopwords

# Globals to be set in main()
LOOKUP: Dict[str, str] = {}
SEEDS: List[str] = []
DOCS = None
MANUAL_TERMS_PATH: str = ''
EVAL_NGRAM = None
MIN_TERMS = 0
# Static parameters
MODEL_DIR: str = ''
OUT_ROOT: str = ''

# Initialize stopwords
STOPWORDS = set(stopwords.words('english'))


def load_lookup(path: str) -> Dict[str, str]:
    """
    Load the JSON lookup map for token normalization.
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_terms(path: str) -> List[str]:
    """
    Load the JSON mapping of category->terms and flatten to a list of seed terms.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    terms: List[str] = []
    for term_list in data.values():
        terms.extend(term_list)
    return terms


def tokenize(text: str) -> List[str]:
    tokens = clean_and_tokenize(text)
    tokens = [LOOKUP.get(t, t) for t in tokens]
    tokens = spellcheck_token_list(tokens)
    return [LOOKUP.get(t, t) for t in tokens]


def embed_phrase(model: Word2Vec, phrase: str) -> np.ndarray | None:
    tokens = clean_and_tokenize(phrase)
    tokens = [LOOKUP.get(t, t) for t in tokens]
    tokens = spellcheck_token_list(tokens)
    tokens = [LOOKUP.get(t, t) for t in tokens]
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def run_one_combo(args_tuple: Tuple[str, int, float]) -> Dict:
    model_file, term_limit, cos_thresh = args_tuple
    # Load model
    model_path = os.path.join(MODEL_DIR, model_file)
    model = Word2Vec.load(model_path)
    model_type = 'cbow' if 'cbow' in model_file else 'skipgram'

    # Build normalized seed vectors
    seed_vecs: List[Tuple[str, np.ndarray]] = []
    for term in SEEDS:
        vec = embed_phrase(model, term)
        if vec is None:
            continue
        norm = np.linalg.norm(vec)
        if norm > 0:
            seed_vecs.append((term, vec / norm))

    # Precompute vocabulary unit vectors
    vocab_words = model.wv.index_to_key
    vocab_mat = model.wv.vectors.astype(np.float32)
    vocab_unit = vocab_mat / np.linalg.norm(vocab_mat, axis=1, keepdims=True)

    # Expand each seed
    exp_map: Dict[str, List[str]] = {}
    for term, unit in seed_vecs:
        sims = vocab_unit @ unit
        idx_sort = np.argsort(-sims)
        if cos_thresh is not None:
            candidates = [i for i in idx_sort if sims[i] >= cos_thresh]
        else:
            candidates = list(idx_sort)
        if len(candidates) < MIN_TERMS:
            sel_idx = idx_sort[:MIN_TERMS]
        else:
            sel_idx = candidates
        sel_idx = sel_idx[:term_limit]
        neighbors = [vocab_words[i] for i in sel_idx]
        exp_map[term] = neighbors

    # Write expansions.json
    combo = f"{model_file.replace('.model','')}_t{term_limit}"
    if cos_thresh is not None:
        combo += f"_c{int(cos_thresh*100)}"
    run_dir = os.path.join(OUT_ROOT, combo)
    os.makedirs(run_dir, exist_ok=True)
    exp_path = os.path.join(run_dir, 'expansions.json')
    with open(exp_path, 'w', encoding='utf-8') as fw:
        json.dump(exp_map, fw, indent=2)

    # Evaluate
    record = {'model': model_type, 'term_limit': term_limit, 'cosine_thresh': cos_thresh}
    if DOCS is not None:
        metrics = evaluate_terms_performance(
            docs=DOCS,
            manual_terms_path=MANUAL_TERMS_PATH,
            expansion_terms_path=exp_path,
            ngram_filter=EVAL_NGRAM,
            tok_fn=tokenize
        )
        record.update(metrics)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parallel combined gridsearch of max-terms and cosine-threshold"
    )
    parser.add_argument('--terms', default="rcpd_terms_6_5.json",
                        help="Path to JSON {category: [terms]} mapping")
    parser.add_argument('--lookup', default="testing/lemma_lookup.json",
                        help="Path to JSON token lookup map")
    parser.add_argument('--model_dir', required=True,
                        help="Directory containing Word2Vec .model files")
    parser.add_argument('--cos_min', type=float, default=None,
                        help="Min cosine threshold")
    parser.add_argument('--cos_max', type=float, default=None,
                        help="Max cosine threshold")
    parser.add_argument('--step_cosine', type=float, default=0.05,
                        help="Step size for cosine grid")
    parser.add_argument('--min_terms', type=int, required=True,
                        help="Minimum expansions per seed")
    parser.add_argument('--max_terms', type=int, required=True,
                        help="Maximum expansions per seed (grid max)")
    parser.add_argument('--manual_dir', default="vocabulary_evaluation/manual_terms_7_12",
                        help="Dir with manual_terms.txt & users.txt for evaluation")
    parser.add_argument('--eval_ngram', type=parse_ngram_filter, default=None,
                        help="Ngram filter (e.g. '<=2')")
    parser.add_argument('--metrics_output', type=str, default=None,
                        help="Path to JSON file for metrics (defaults to out_root/metrics.json)")
    parser.add_argument('--out_root', default=None,
                        help="Base directory for output expansions and metrics")
    args = parser.parse_args()

    # Set globals
    LOOKUP = load_lookup(args.lookup)
    SEEDS = load_terms(args.terms)
    MODEL_DIR = args.model_dir
    MIN_TERMS = args.min_terms
    MAX_TERMS = args.max_terms

    # Prepare evaluation docs
    if args.manual_dir:
        MANUAL_TERMS_PATH = os.path.join(args.manual_dir, 'manual_terms.txt')
        users = load_user_list(os.path.join(args.manual_dir, 'users.txt'))
        DOCS = return_documents(
            db_name='reddit', collection_name='noburp_all',
            filter_subreddits=['noburp'], filter_users=users
        )
    else:
        DOCS = None
        MANUAL_TERMS_PATH = ''

    EVAL_NGRAM = args.eval_ngram

    # Build grid parameters
    term_limits = list(range(args.min_terms, args.max_terms+1, 5))
    if args.cos_min is not None and args.cos_max is not None:
        n_cos = int(round((args.cos_max - args.cos_min) / args.step_cosine)) + 1
        cos_values = np.linspace(args.cos_min, args.cos_max, n_cos)
    else:
        cos_values = [None]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_ROOT = args.out_root or os.path.join(MODEL_DIR, f"combined_grid_{timestamp}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Prepare tasks: one tuple per (model_file, term_limit, cos_thresh)
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.model')]
    tasks = [
        (mf, tl, ct)
        for mf in model_files
        for tl in term_limits
        for ct in cos_values
    ]

    # Run in parallel
    all_metrics: List[Dict] = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_one_combo, tasks):
            if result is not None:
                all_metrics.append(result)

    # Write metrics
    metrics_file = args.metrics_output or os.path.join(OUT_ROOT, 'metrics.json')
    existing = []
    if os.path.exists(metrics_file):
        existing = json.load(open(metrics_file, 'r', encoding='utf-8'))
    existing.extend(all_metrics)
    with open(metrics_file, 'w', encoding='utf-8') as mf:
        json.dump(existing, mf, indent=2)
    print(f"Saved {len(all_metrics)} metric records to {metrics_file}")
    print("Gridsearch complete!")
