import os
import sys
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
from gensim.models import Word2Vec
from datetime import datetime
from tqdm import tqdm

sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from query_mongo import return_documents
from spellchecker_folder.spellchecker import spellcheck_token_list
from evaluate_expansions_lemmatized import evaluate_terms_performance, load_user_list, parse_ngram_filter
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def load_terms(path: str, tok_fn=None) -> Dict[str, List[str]]:
    def clean_phrase(phrase: str) -> str:
        tokens = tok_fn(phrase.replace('_', ' '))
        return ' '.join(tokens)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # orig_map: category -> List[terms]
    return {clean_phrase(cat): [clean_phrase(t) for t in terms] for cat, terms in data.items()}


def load_lookup(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_frequent_ngrams(max_ngram: int, tok_fn, lookup_map: dict) -> List[str]:
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        mongo_uri="mongodb://localhost:27017/"
    )
    ngrams = set()
    for doc in tqdm(docs, desc="Loading docs for ngrams"):
        tokens = [lookup_map.get(t, t) for t in tok_fn(doc)]
        L = len(tokens)
        for n in range(2, max_ngram + 1):
            if L < n:
                break
            for i in range(L - n + 1):
                ngrams.add(" ".join(tokens[i:i + n]))
    return list(ngrams)


def embed_phrase(model: Word2Vec, phrase: str, tok_fn, lookup_map: dict) -> np.ndarray | None:
    tokens = [lookup_map.get(t, t) for t in tok_fn(phrase)]
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        print(f"Warning: no valid tokens for phrase '{phrase}'")
        return None
    return np.mean(vecs, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gridsearch term-level neighbors")
    parser.add_argument('--seed_terms',      default="rcpd_terms_6_5.json", help="Path to JSON with original terms map")
    parser.add_argument('--lookup',     default="testing/lemma_lookup.json", help="Path to JSON lookup for normalization")
    parser.add_argument('--model_dir',  required=True)
    parser.add_argument('--kmin',       type=int, help="Min number of neighbors to retrieve")
    parser.add_argument('--kmax',       type=int, help="Max number of neighbors to retrieve")
    parser.add_argument('--cos_min',    type=float, help="Min cosine threshold")
    parser.add_argument('--cos_max',    type=float, help="Max cosine threshold")
    parser.add_argument('--manual_dir', default="vocabulary_evaluation/manual_terms_7_12", help="Dir with manual_terms.txt and users.txt for evaluation")
    parser.add_argument('--eval_ngram', type=parse_ngram_filter, default=None,
                        help="Ngram filter (e.g. '<=2')")
    parser.add_argument('--metrics_output', type=str,
                        help="Path to JSON file for metrics (defaults to out_root/metrics.json)")
    parser.add_argument('--out_root',   default=None)
    args = parser.parse_args()

    lookup   = load_lookup(args.lookup)

    def tok_fn(text: str) -> List[str]:
        tokens = clean_and_tokenize(text)
        tokens = [lookup.get(t, t) for t in tokens]
        tokens = spellcheck_token_list(tokens)
        tokens = [lookup.get(t, t) for t in tokens]
        return tokens

    orig_map = load_terms(args.seed_terms, tok_fn=tok_fn)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_root  = args.out_root or os.path.join(args.model_dir, f"term_grid_{timestamp}")
    os.makedirs(out_root, exist_ok=True)
    if not args.metrics_output:
        args.metrics_output = os.path.join(out_root, 'metrics.json')

    # Pre-load manual evaluation docs once
    if args.manual_dir:
        manual_terms_path = os.path.join(args.manual_dir, 'manual_terms.txt')
        users = load_user_list(os.path.join(args.manual_dir, 'users.txt'))
        docs = return_documents(
            db_name='reddit',
            collection_name='noburp_all',
            filter_subreddits=['noburp'],
            filter_users=users
        )
    else:
        docs = None

    all_metrics: List[Dict] = []

    for model_file in ['word2vec_skipgram.model', 'word2vec_cbow.model']:
        mpath = os.path.join(args.model_dir, model_file)
        if not os.path.exists(mpath):
            continue
        model = Word2Vec.load(mpath)
        model_type = 'cbow' if 'cbow' in model_file else 'skipgram'

        # Build term-level unit vectors once
        term_vecs: List[Tuple[str, np.ndarray]] = []
        for terms in orig_map.values():
            for term in terms:
                vec = embed_phrase(model, term, tok_fn, lookup)
                if vec is None:
                    continue
                norm = np.linalg.norm(vec)
                if norm > 0:
                    term_vecs.append((term, vec / norm))

        # Precompute static structures
        vocab_words = model.wv.index_to_key
        vocab_mat   = model.wv.vectors.astype(np.float32)
        vocab_unit  = vocab_mat / np.linalg.norm(vocab_mat, axis=1, keepdims=True)
        if args.eval_ngram:
            max_n = args.eval_ngram[1]
            frequent_ngrams = extract_frequent_ngrams(max_n, tok_fn, lookup)
        else:
            frequent_ngrams = []
        ngram_vecs = []
        for ng in tqdm(frequent_ngrams, desc="Embedding ngrams"):
            v = embed_phrase(model, ng, tok_fn, lookup)
            if v is not None:
                n = np.linalg.norm(v)
                ngram_vecs.append(v / n if n > 0 else None)
            else:
                ngram_vecs.append(None)
        valid_idx     = [i for i,v in enumerate(ngram_vecs) if v is not None]
        ngram_phrases = [frequent_ngrams[i] for i in valid_idx]
        ngram_matrix  = np.vstack([ngram_vecs[i] for i in valid_idx]) if valid_idx else np.zeros((0, vocab_mat.shape[1]), dtype=np.float32)

        # Efficient top-K gridsearch
        if args.kmin is not None and args.kmax is not None:
            k_values = list(range(args.kmin, args.kmax + 1, 5))
            max_k = max(k_values)
            exp_maps: Dict[int, Dict[str, List[str]]] = {k: {} for k in k_values}
            for term, unit in term_vecs:
                # get top max_k unigrams once
                uni_max = model.wv.similar_by_vector(unit, topn=max_k)
                uni_terms = [w for w,_ in uni_max if w not in STOPWORDS]
                # get top max_k ngrams once
                sims_ng = ngram_matrix @ unit
                sorted_ng = np.argsort(-sims_ng)
                for k in k_values:
                    uni_hits = set(uni_terms[:k])
                    ng_hits = {ngram_phrases[i] for i in sorted_ng[:k]}
                    exp_maps[k][term] = list(uni_hits | ng_hits)

            # write and evaluate per k
            for k, exp_map in exp_maps.items():
                combo = f"{model_file.replace('.model','')}_k{k}"
                run_dir = os.path.join(out_root, combo)
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, 'expansions.json')
                with open(out_path, 'w', encoding='utf-8') as fw:
                    json.dump(exp_map, fw, indent=2)
                print(f"Wrote {combo} → terms: {len(exp_map)}")

                if docs is not None:
                    metrics = evaluate_terms_performance(
                        docs=docs,
                        manual_terms_path=manual_terms_path,
                        expansion_terms_path=out_path,
                        ngram_filter=args.eval_ngram,
                        tok_fn=tok_fn
                    )
                    record = {'model': model_type, 'topk': k, **metrics}
                    all_metrics.append(record)

        # Efficient cosine-threshold gridsearch
        elif args.cos_min is not None and args.cos_max is not None:
            n_steps    = int(round((args.cos_max - args.cos_min) / 0.05)) + 1
            cos_values = np.linspace(args.cos_min, args.cos_max, n_steps)
            exp_maps: Dict[float, Dict[str, List[str]]] = {c: {} for c in cos_values}
            for term, unit in term_vecs:
                sims_u  = vocab_unit @ unit
                sims_ng = ngram_matrix @ unit
                for c in cos_values:
                    uni_hits = {vocab_words[i] for i,v in enumerate(sims_u) if v >= c and vocab_words[i] not in STOPWORDS}
                    ng_hits  = {ngram_phrases[i] for i,v in enumerate(sims_ng) if v >= c}
                    exp_maps[c][term] = list(uni_hits | ng_hits)

            # write and evaluate per threshold
            for c, exp_map in exp_maps.items():
                combo = f"{model_file.replace('.model','')}_cos{int(c*100)}"
                run_dir = os.path.join(out_root, combo)
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, 'expansions.json')
                with open(out_path, 'w', encoding='utf-8') as fw:
                    json.dump(exp_map, fw, indent=2)
                print(f"Wrote {combo} → terms: {len(exp_map)}")

                if docs is not None:
                    metrics = evaluate_terms_performance(
                        docs=docs,
                        manual_terms_path=manual_terms_path,
                        expansion_terms_path=out_path,
                        ngram_filter=args.eval_ngram,
                        tok_fn=tok_fn
                    )
                    record = {'model': model_type, 'cos': c, **metrics}
                    all_metrics.append(record)

        else:
            print("Error: specify either kmin/kmax or cos_min/cos_max for gridsearch.")
            sys.exit(1)

    # write all metrics
    try:
        existing = json.load(open(args.metrics_output, 'r', encoding='utf-8'))
    except FileNotFoundError:
        existing = []
    existing.extend(all_metrics)
    with open(args.metrics_output, 'w', encoding='utf-8') as mf:
        json.dump(existing, mf, indent=2)
    print(f"→ Appended {len(all_metrics)} records to {args.metrics_output}")
    print("Gridsearch complete!")
