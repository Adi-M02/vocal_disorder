# outline_pipeline.py
# Modular pipeline for BERT‑based vocabulary expansion with two‑phase extraction, gridsearch, and evaluation

import argparse
import json
import os
import itertools
import sys 
import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from word2vec_expansion.evaluate_expansions_lemmatized import evaluate_terms_performance, load_user_list

# Ensure NLTK stopwords are available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words("english"))


def compute_centroids(cat_json_path, model, tokenizer, device):
    """
    Load category -> term lists from JSON and compute centroid for each category.
    JSON format: {"cat1": ["term1", ...], ...}
    """
    with open(cat_json_path, 'r', encoding='utf-8') as f:
        cat_terms = json.load(f)
    centroids = {}
    for cat, terms in cat_terms.items():
        embs = []
        for term in terms:
            inputs = tokenizer(term, return_tensors='pt', truncation=True).to(device)
            with torch.no_grad(): out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            embs.append(emb)
        centroids[cat] = torch.stack(embs).mean(dim=0)
    return centroids


def extract_doc_candidates(centroids, model, tokenizer, device):
    """
    Process all documents once, extract 1/2/3‑grams, filter n‑grams that start/end with stopwords, compute cosine sims to centroids.
    Returns list of dicts: {phrase, category, count, avg_score} and list of raw documents.
    """
    # Load and clean documents
    raw = return_documents("reddit", "noburp_all", ["noburp"])
    texts = [str(doc) for doc in raw if doc]

    stats = defaultdict(lambda: {'count': 0, 'scores': []})
    for text in texts:
        tokens = clean_and_tokenize(text)
        if not tokens:
            continue
        joined = tokenizer.convert_tokens_to_string(tokens)
        inputs = tokenizer(joined, return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        embeddings = out.last_hidden_state.squeeze(0).cpu()

        # extract n-grams and skip those starting or ending with a stopword
        for n in (1, 2, 3):
            for i in range(len(tokens) - n + 1):
                ngram_tokens = tokens[i:i+n]
                # skip if starts or ends with stopword
                if ngram_tokens[0].lower() in STOPWORDS or ngram_tokens[-1].lower() in STOPWORDS:
                    continue
                phrase = " ".join(ngram_tokens)
                emb_ng = embeddings[i:i+n].mean(dim=0)
                for cat, centroid in centroids.items():
                    sim = F.cosine_similarity(
                        emb_ng.unsqueeze(0), centroid.unsqueeze(0)
                    ).item()
                    key = (phrase, cat)
                    stats[key]['count'] += 1
                    stats[key]['scores'].append(sim)

    candidates = []
    for (phrase, cat), v in stats.items():
        avg_score = sum(v['scores']) / len(v['scores'])
        candidates.append({'phrase': phrase, 'category': cat,
                           'count': v['count'], 'avg_score': avg_score})
    return candidates, texts


def filter_candidates(candidates, sim_thresh, freq_thresh):
    """
    Filter candidate dicts by similarity & frequency thresholds.
    """
    return [c for c in candidates if c['avg_score'] >= sim_thresh and c['count'] >= freq_thresh]


def main():
    parser = argparse.ArgumentParser(
        description='BERT vocab expansion gridsearch + evaluation'
    )
    parser.add_argument('--categories',     required=True, help='JSON of category -> terms')
    parser.add_argument('--manual_dir',   required=True, help='Manual terms directory')
    parser.add_argument('--model_dir',      required=True, help='Fine-tuned BERT directory')
    parser.add_argument('--sim_min',    type=float, required=True, help='Min cosine sim')
    parser.add_argument('--sim_max',    type=float, required=True, help='Max cosine sim')
    parser.add_argument('--sim_step',   type=float, default=0.02)
    parser.add_argument('--freq_min',   type=float,   required=True, help='Min frequency')
    parser.add_argument('--freq_max',   type=float,   required=True, help='Max frequency')
    parser.add_argument('--freq_step',  type=float,   default=0.01)
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available(): device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        raise RuntimeError("No supported GPU (CUDA or MPS) available.")

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model     = AutoModel.from_pretrained(args.model_dir).to(device)

    # Phase 1: compute centroids & candidates
    centroids, docs = None, None
    centroids = compute_centroids(args.categories, model, tokenizer, device)
    candidates, docs = extract_doc_candidates(centroids, model, tokenizer, device)

    # Build grids
    sim_vals = []
    cur = args.sim_min
    while cur <= args.sim_max + 1e-8:
        sim_vals.append(round(cur, 6)); cur += args.sim_step
    freq_vals = np.arange(args.freq_min, args.freq_max + 1e-8, args.freq_step).tolist()

    # Create root folder
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, 'gridsearch', ts)
    os.makedirs(root_dir, exist_ok=True)

    all_metrics = []
    # Gridsearch
    for sim_t, freq_t in itertools.product(sim_vals, freq_vals):
        run_name = f"result_sim{sim_t}_freq{freq_t}"
        run_dir  = os.path.join(root_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Filter expansions
        filtered = filter_candidates(candidates, sim_t, freq_t)
        # Group by category
        expansions = defaultdict(list)
        for c in filtered:
            expansions[c['category']].append(c['phrase'])
        # Write expansions.json
        exp_path = os.path.join(run_dir, 'expansions.json')
        with open(exp_path, 'w', encoding='utf-8') as ew:
            json.dump(dict(expansions), ew, indent=2)
        manual_terms_path = os.path.join(args.manual_dir,'manual_terms.txt')
        users_file = os.path.join(args.manual_dir,'users.txt')    
        users = load_user_list(users_file)
        manual_docs = return_documents(db_name='reddit',collection_name='noburp_all',filter_subreddits=['noburp'],filter_users=users)   
        # Evaluate
        metrics = evaluate_terms_performance(
            docs=manual_docs,
            manual_terms_path=manual_terms_path,
            expansion_terms_path=exp_path,
            ngram_filter=None,
            tok_fn=clean_and_tokenize,
            lemmatize=False,
            lemma_map=None
        )
        # Write evaluation.txt
        eval_path = os.path.join(run_dir, 'evaluation.txt')
        with open(eval_path, 'w', encoding='utf-8') as ef:
            for k,v in metrics.items(): ef.write(f"{k}: {v}\n")

        # Record metrics
        record = {'sim': sim_t, 'freq': freq_t}
        record.update(metrics)
        all_metrics.append(record)
        print(f"Completed {run_name} → {len(filtered)} phrases, metrics: {metrics}")

    # Save all metrics
    mpath = os.path.join(root_dir, 'metrics.json')
    with open(mpath, 'w', encoding='utf-8') as mf:
        json.dump(all_metrics, mf, indent=2)
    print(f"Gridsearch complete. Metrics written to {mpath}")


if __name__ == '__main__':
    main()
