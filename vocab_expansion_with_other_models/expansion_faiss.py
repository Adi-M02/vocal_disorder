# outline_pipeline.py
# Modular pipeline for BERT‑based vocabulary expansion with FAISS similarity, batching, logging, sampling, gridsearch, and evaluation

import argparse
import json
import os
import itertools
import sys
import datetime
import logging
import math
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import nltk
from nltk.corpus import stopwords

# Add project path
sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from word2vec_expansion.evaluate_expansions_lemmatized import (
    evaluate_terms_performance,
    load_user_list
)

# Configure logging
def setup_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S'
    )

# Ensure NLTK stopwords are available
def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

STOPWORDS = get_stopwords()


def compute_centroids(cat_json_path, model, tokenizer, device):
    logging.info('Computing centroids from category JSON: %s', cat_json_path)
    with open(cat_json_path, 'r', encoding='utf-8') as f:
        cat_terms = json.load(f)

    centroids = {}
    for cat, terms in cat_terms.items():
        logging.info('  Category `%s`: %d seed terms', cat, len(terms))
        embs = []
        for term in terms:
            text = f"{cat} {term}"
            inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            embs.append(emb.numpy().astype('float32'))
        centroids[cat] = np.stack(embs).mean(axis=0)
        logging.info('    Centroid for `%s` computed', cat)

    logging.info('Finished computing centroids')
    return centroids


def build_faiss_index(centroids):
    # Prepare name list and matrix
    cat_names = list(centroids.keys())
    mat = np.stack([centroids[n] for n in cat_names])
    # Normalize
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    dim = mat.shape[1]
    # GPU resources and index
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, dim)
    index.add(mat)
    return index, cat_names


def extract_doc_candidates(index, cat_names, model, tokenizer, device,
                            sample_size=5000, batch_size=8):
    logging.info('Loading documents from MongoDB')
    raw = return_documents('reddit', 'noburp_all', ['noburp'])
    all_texts = [str(doc) for doc in raw if doc]
    total_docs = len(all_texts)
    logging.info('Retrieved %d documents', total_docs)
    if sample_size > 0 and total_docs > sample_size:
        texts = random.sample(all_texts, sample_size)
        logging.info('Sampled %d documents for extraction', sample_size)
    else:
        texts = all_texts

    stats = defaultdict(lambda: {'count': 0, 'scores': []})
    num_batches = math.ceil(len(texts) / batch_size)

    for bidx in range(num_batches):
        start = bidx * batch_size
        batch_texts = texts[start:start + batch_size]
        logging.info('Processing batch %d/%d', bidx + 1, num_batches)

        tok_lists = [clean_and_tokenize(t) for t in batch_texts]
        tok_lists = [t for t in tok_lists if t]
        if not tok_lists:
            continue
        logging.info('  %d/%d docs have tokens', len(tok_lists), len(batch_texts))

        joined = [tokenizer.convert_tokens_to_string(t) for t in tok_lists]
        inputs = tokenizer(
            joined,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8196
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        batch_embs = out.last_hidden_state.cpu().numpy()  # [B, L, H]

        for doc_i, toks in enumerate(tok_lists):
            emb = batch_embs[doc_i]
            for n in (1, 2, 3):
                for i in range(len(toks) - n + 1):
                    ngram = toks[i:i + n]
                    if ngram[0].lower() in STOPWORDS or ngram[-1].lower() in STOPWORDS:
                        continue
                    phrase = ' '.join(ngram)
                    sub = emb[i:i + n].mean(axis=0)
                    sub /= np.linalg.norm(sub) + 1e-12
                    # search all centroids
                    D, I = index.search(sub.reshape(1, -1), len(cat_names))
                    for score, idx in zip(D[0], I[0]):
                        cat = cat_names[idx]
                        key = (phrase, cat)
                        stats[key]['count'] += 1
                        stats[key]['scores'].append(float(score))

    candidates = []
    for (phrase, cat), v in stats.items():
        avg_score = sum(v['scores']) / len(v['scores'])
        candidates.append({
            'phrase': phrase,
            'category': cat,
            'count': v['count'],
            'avg_score': avg_score
        })
    logging.info('Extraction complete: %d unique phrase-category pairs', len(candidates))
    return candidates, texts


def filter_candidates(candidates, sim_thresh):
    logging.info('Filtering candidates with sim >= %.3f', sim_thresh)
    filtered = [c for c in candidates if c['avg_score'] >= sim_thresh]
    logging.info('  %d candidates passed filtering', len(filtered))
    return filtered


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description='BERT vocab expansion pipeline'
    )
    parser.add_argument('--categories',   required=True, help='JSON of category->terms')
    parser.add_argument('--manual_dir',   required=True, help='Dir with manual_terms.txt and users.txt')
    parser.add_argument('--model_dir',    required=True, help='Fine-tuned BERT model dir')
    parser.add_argument('--sim_min',   type=float, required=True, help='Min similarity threshold')
    parser.add_argument('--sim_max',   type=float, required=True, help='Max similarity threshold')
    parser.add_argument('--sim_step',  type=float, default=0.01, help='Step size for sim grid')
    parser.add_argument('--sample_size', type=int, default=5000, help='Number of docs to sample, -1 for all docs')
    parser.add_argument('--batch_size',  type=int, default=8, help='Batch size for BERT inference')
    parser.add_argument('--del_cache', action='store_true', help='Delete cache if exists')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        raise RuntimeError('No supported GPU available')
    logging.info('Using device: %s', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir).to(device)

    centroids = compute_centroids(args.categories, model, tokenizer, device)
    index, cat_names = build_faiss_index(centroids)

    cache_path = os.path.join(args.model_dir, 'extraction_cache.pkl')
    if args.del_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        logging.info('Deleted existing cache at %s', cache_path)

    if os.path.exists(cache_path):
        logging.info('Loading cached candidates/docs from %s', cache_path)
        with open(cache_path, 'rb') as f:
            candidates, docs = pickle.load(f)
    else:
        candidates, docs = extract_doc_candidates(
            index, cat_names, model, tokenizer, device,
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
        with open(cache_path, 'wb') as f:
            pickle.dump((candidates, docs), f)
        logging.info('Cached extraction to %s', cache_path)

    stats_counts = defaultdict(dict)
    for c in candidates:
        stats_counts[c['category']][c['phrase']] = c['count']
    stats_path = os.path.join(args.model_dir, 'stats_counts.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_counts, f, indent=2)
    logging.info('Saved stats_counts to %s', stats_path)

    sim_vals = []
    cur = args.sim_min
    while cur <= args.sim_max + 1e-8:
        sim_vals.append(round(cur, 6))
        cur += args.sim_step

    users = load_user_list(os.path.join(args.manual_dir, 'users.txt'))
    manual_docs = return_documents(
        db_name='reddit', collection_name='noburp_all',
        filter_subreddits=['noburp'], filter_users=users
    )
    all_metrics = []

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join(os.path.dirname(__file__), f'gridsearch_{timestamp}')
    os.makedirs(out_root, exist_ok=True)

    for sim_t in sim_vals:
        run_name = f'result_sim_{sim_t}'
        run_dir = os.path.join(out_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        logging.info('Running grid step: %s', run_name)

        filtered = filter_candidates(candidates, sim_t)
        expansions = defaultdict(list)
        for c in filtered:
            expansions[c['category']].append(c['phrase'])

        exp_path = os.path.join(run_dir, 'expansions.json')
        with open(exp_path, 'w', encoding='utf-8') as f:
            json.dump(dict(expansions), f, indent=2)

        metrics = evaluate_terms_performance(
            docs=manual_docs,
            manual_terms_path=os.path.join(args.manual_dir, 'manual_terms.txt'),
            expansion_terms_path=exp_path,
            ngram_filter=None,
            tok_fn=clean_and_tokenize,
            lemmatize=False,
            lemma_map=None
        )
        eval_path = os.path.join(run_dir, 'evaluation.txt')
        with open(eval_path, 'w', encoding='utf-8') as f:
            for k, v in metrics.items():
                f.write(f'{k}: {v}\n')

        record = {'sim': float(sim_t)}
        record.update(metrics)
        all_metrics.append(record)
        logging.info('Completed %s → %d phrases', run_name, len(filtered))

    metrics_path = os.path.join(out_root, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    logging.info('Gridsearch complete. Metrics at: %s', metrics_path)

if __name__ == '__main__':
    main()
