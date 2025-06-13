import os
import sys
import json
import math
import argparse
import itertools
from collections import Counter
from typing import List, Dict

import numpy as np
from gensim.models import Word2Vec
from datetime import datetime
from tqdm import tqdm

# Add your project paths here
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from query_mongo import return_documents
from spellchecker_folder.spellchecker import spellcheck_token_list
from evaluate_expansions_lemmatized import evaluate_terms_performance, load_user_list, parse_ngram_filter
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def load_terms(path: str, tok_fn=None, lookup_map=None) -> Dict[str, list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    for cat, terms in data.items():
        cat_clean = cat.replace('_', ' ')
        if tok_fn and lookup_map is not None:
            cat_tokens = [lookup_map.get(t, t) for t in tok_fn(cat_clean)]
            cat_clean = ' '.join(cat_tokens)
            terms_clean = []
            for t in terms:
                t_clean = t.replace('_', ' ')
                t_tokens = [lookup_map.get(tok, tok) for tok in tok_fn(t_clean)]
                terms_clean.append(' '.join(t_tokens))
        else:
            terms_clean = [t.replace('_', ' ') for t in terms]
        result[cat_clean] = terms_clean
    return result


def load_lookup(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def extract_frequent_ngrams(
    max_ngram: int,
    tok_fn,
    lookup_map: dict
) -> list[str]:
    # fetch all documents
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        mongo_uri="mongodb://localhost:27017/"
    )
    
    counts = Counter()
    
    # slide an n-length window over each doc’s token list
    for doc in tqdm(docs, desc=f"Loading docs for ngrams"):
        tokens = [lookup_map.get(t, t) for t in tok_fn(doc)]
        L = len(tokens)
        for n in range(2, max_ngram + 1):
            if L < n:
                break
            for i in range(L - n + 1):
                gram = tuple(tokens[i:i + n])
                # skip if first or last token is a stopword
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                counts[gram] += 1

    # hardcoded min_count for 2-gram and 3-gram
    result = []
    for gram, cnt in counts.items():
        n = len(gram)
        if n == 2 and cnt >= 5:
            result.append(" ".join(gram))
        elif n == 3 and cnt >= 3:
            result.append(" ".join(gram))
    return result


def embed_phrase(model: Word2Vec, phrase: str, tok_fn, lookup_map: dict) -> np.ndarray | None:
    tokens = [lookup_map.get(t,t) for t in tok_fn(phrase)]
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vecs:
        print(f"Warning: No valid tokens found for phrase '{phrase}'")
        return None
    return np.mean(vecs, axis=0)


def compute_triplets(model: Word2Vec,
                     terms_map: Dict[str,list[str]],
                     tok_fn,
                     lookup_map: dict) -> tuple[list[np.ndarray], list[str]]:
    vecs, cats = [], []
    for category, terms in terms_map.items():
        cat_vec = embed_phrase(model, category, tok_fn, lookup_map)
        if cat_vec is None:
            print(f"Warning: Category '{category}' has no valid embedding.")
            continue
        for t1, t2 in itertools.combinations(terms, 2):
            v1 = embed_phrase(model, t1, tok_fn, lookup_map)
            v2 = embed_phrase(model, t2, tok_fn, lookup_map)
            if v1 is None or v2 is None:
                continue
            trip = (v1 + v2 + cat_vec) / 3.0
            norm = np.linalg.norm(trip)
            if norm>0:
                vecs.append(trip / norm)
                cats.append(category)
    return vecs, cats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Gridsearch Word2Vec expansion"
    )
    parser.add_argument('--terms',         required=True)
    parser.add_argument('--model_dir',     required=True)
    parser.add_argument('--lookup',        default='testing/lemma_lookup.json')
    parser.add_argument('--spellcheck',    action='store_true')
    parser.add_argument('--kmin', default=None, type=int)
    parser.add_argument('--kmax', default=None, type=int)
    parser.add_argument('--cos_min',      type=float, default=None)
    parser.add_argument('--cos_max',      type=float, default=None)
    parser.add_argument('--freq_min',      type=float, required=True, help="Minimum freq threshold (step=0.01)")
    parser.add_argument('--freq_max',      type=float, required=True)
    # parser.add_argument('--ngram_min',    type=int, default=5,
    #                     help="Min count for ngram inclusion")
    parser.add_argument('--manual_dir',    help="Dir with manual_terms.txt and users.txt for evaluation")
    parser.add_argument('--eval_ngram',    type=parse_ngram_filter, default=None, help="Ngram filter (e.g. '<=2')")
    parser.add_argument('--metrics_output', type=str,
                        help="Path to JSON file for metrics (defaults to out_root/metrics.json)")
    parser.add_argument('--out_root',      default=None,
                        help="Directory for outputs")
    args = parser.parse_args()

    if args.spellcheck:
        def tok_fn(text):
            return spellcheck_token_list(clean_and_tokenize(text))
    else:
        tok_fn = clean_and_tokenize
    lookup = load_lookup(args.lookup)
    orig_map = load_terms(args.terms, lookup_map=lookup)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_root = args.out_root or os.path.join(args.model_dir, f"grid_{timestamp}")
    os.makedirs(out_root, exist_ok=True)
    if not args.metrics_output:
        args.metrics_output = os.path.join(out_root, 'metrics.json')
    # Write run arguments to info.json in out_root
    info_path = os.path.join(out_root, 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f_info:
        json.dump(vars(args), f_info, indent=2)

    # prepare grid
    if args.kmin and args.kmax:
        k_values = list(range(args.kmin, args.kmax+1, 5))
    elif args.cos_min and args.cos_max:
        cos_values = np.arange(args.cos_min, args.cos_max + 0.01, 0.05)
    freq_values = list(np.arange(args.freq_min, args.freq_max+1e-8, 0.01))
    all_metrics: List[Dict] = []

    for model_file in ['word2vec_cbow.model','word2vec_skipgram.model']:
        mpath = os.path.join(args.model_dir, model_file)
        if not os.path.exists(mpath):
            continue
        model = Word2Vec.load(mpath)

        # precompute structures
        triplet_vecs, triplet_cats = compute_triplets(model, orig_map, tok_fn, lookup)

        vocab_words = model.wv.index_to_key
        vocab_mat = model.wv.vectors.astype(np.float32)
        vocab_unit = vocab_mat / np.linalg.norm(vocab_mat, axis=1, keepdims=True)

        frequent_ngrams = extract_frequent_ngrams(args.eval_ngram[1], tok_fn, lookup)
        # embed ngrams
        ngram_vecs = []
        for ng in frequent_ngrams:
            v = embed_phrase(model, ng, tok_fn, lookup)
            if v is not None:
                n = np.linalg.norm(v)
                if n>0:
                    ngram_vecs.append(v/n)
                else:
                    ngram_vecs.append(None)
            else:
                ngram_vecs.append(None)
        valid_idx = [i for i,v in enumerate(ngram_vecs) if v is not None]
        ngram_phrases = [frequent_ngrams[i] for i in valid_idx]
        ngram_matrix = np.vstack([ngram_vecs[i] for i in valid_idx]) if valid_idx else np.zeros((0, vocab_mat.shape[1]), dtype=np.float32)
        # k nearest neighbors gridsearch
        if args.kmin and args.kmax:
            for topk, freq in itertools.product(k_values, freq_values):
                # expansion logic
                per_cat = {c: [] for c in orig_map}
                for vec, cat in zip(triplet_vecs, triplet_cats):
                    n = np.linalg.norm(vec)
                    if n==0:
                        continue
                    unit = vec / n
                    # unigram
                    uni_hits = {w for w,_ in model.wv.similar_by_vector(unit, topn=topk)}
                    uni_hits = {w for w in uni_hits if w not in STOPWORDS}
                    # ngram
                    sims_ng = ngram_matrix @ unit
                    topn = np.argsort(-sims_ng)[:topk]
                    ng_hits = {ngram_phrases[i] for i in topn}
                    per_cat[cat].append(uni_hits | ng_hits)
                # freq filter
                exp_map = {}
                for cat, subsets in per_cat.items():
                    q = len(subsets)
                    cnt = Counter(itertools.chain.from_iterable(subsets))
                    need = math.ceil(freq * q)
                    exp_map[cat] = [t for t,c in cnt.items() if c>=need]
                # write expansion
                run_map = {cat: orig_map[cat] + exp_map.get(cat, []) for cat in orig_map}
                combo = f"{model_file.replace('.model','')}_k{topk}_f{int(freq * 100)}"
                run_dir = os.path.join(out_root, combo)
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, 'expansions.json')
                with open(out_path,'w',encoding='utf-8') as fw:
                    json.dump(run_map, fw, indent=2)
                print(f"Wrote {combo} → categories: {len(run_map)}")
                # evaluation
                if args.manual_dir:
                    from evaluate_expansions_lemmatized import evaluate_terms_performance, load_user_list, parse_ngram_filter
                    manual_terms = os.path.join(args.manual_dir,'manual_terms.txt')
                    users_file = os.path.join(args.manual_dir,'users.txt')
                    ngram_filter = args.eval_ngram
                    users = load_user_list(users_file)
                    docs = return_documents(db_name='reddit',collection_name='noburp_all',filter_subreddits=['noburp'],filter_users=users)
                    metrics = evaluate_terms_performance(docs=docs,manual_terms_path=manual_terms,expansion_terms_path=out_path,ngram_filter=ngram_filter,tok_fn=tok_fn, lemmatize=True, lemma_map=lookup)
                    model_type = 'cbow' if 'cbow' in model_file else 'skipgram'
                    record = {'model':model_type,'topk':topk,'freq_threshold':freq,**metrics}
                    all_metrics.append(record)
        #cosine similarity gridsearch
        elif args.cos_min and args.cos_max:
            for cos_thresh, freq in itertools.product(cos_values, freq_values):
                per_cat = {c: [] for c in orig_map}
                for vec, cat in zip(triplet_vecs, triplet_cats):
                    norm = np.linalg.norm(vec)
                    if norm == 0:
                        continue
                    unit = vec / norm

                    # unigram by cosine threshold
                    sims_u  = vocab_unit @ unit                   # shape (V,)
                    uni_hits = {
                        vocab_words[i]
                        for i, v in enumerate(sims_u)
                        if v >= cos_thresh and vocab_words[i] not in STOPWORDS
                    }

                    # n‐gram by cosine threshold
                    sims_ng  = ngram_matrix @ unit                 # shape (N,)
                    ng_hits  = {
                        ngram_phrases[i]
                        for i, v in enumerate(sims_ng)
                        if v >= cos_thresh
                    }

                    per_cat[cat].append(uni_hits | ng_hits)

                # frequency filter (unchanged)
                exp_map = {}
                for cat, subsets in per_cat.items():
                    q    = len(subsets)
                    cnt  = Counter(itertools.chain.from_iterable(subsets))
                    need = math.ceil(freq * q)
                    exp_map[cat] = [t for t, c in cnt.items() if c >= need]

                # write expansion JSON
                run_map = {cat: orig_map[cat] + exp_map.get(cat, []) for cat in orig_map}
                combo   = f"{model_file.replace('.model','')}_cos{int(cos_thresh * 100)}_f{int(freq * 100)}"
                run_dir = os.path.join(out_root, combo)
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, 'expansions.json')
                with open(out_path, 'w', encoding='utf-8') as fw:
                    json.dump(run_map, fw, indent=2)
                print(f"Wrote {combo} → categories: {len(run_map)}")

                # evaluation (unchanged, but record 'cos' instead of 'topk')
                if args.manual_dir:
                    users = load_user_list(os.path.join(args.manual_dir, 'users.txt'))
                    docs  = return_documents(
                        db_name='reddit',
                        collection_name='noburp_all',
                        filter_subreddits=['noburp'],
                        filter_users=users
                    )
                    metrics = evaluate_terms_performance(
                        docs=docs,
                        manual_terms_path=os.path.join(args.manual_dir,'manual_terms.txt'),
                        expansion_terms_path=out_path,
                        ngram_filter=args.eval_ngram,
                        tok_fn=tok_fn,
                        lemmatize=True,
                        lemma_map=lookup
                    )
                    model_type = 'cbow' if 'cbow' in model_file else 'skipgram'
                    record = {
                        'model': model_type,
                        'cos': cos_thresh,
                        'freq_threshold': freq,
                        **metrics
                    }
                    all_metrics.append(record)
        else:
            print("Error: Must specify either kmin/kmax or cos_min/cos_max for gridsearch.")
            sys.exit(1)

    # write metrics
    try:
        existing = json.load(open(args.metrics_output,'r',encoding='utf-8'))
    except FileNotFoundError:
        existing=[]
    existing.extend(all_metrics)
    with open(args.metrics_output,'w',encoding='utf-8') as mf:
        json.dump(existing,mf,indent=2)
    print(f"→ Appended {len(all_metrics)} records to {args.metrics_output}")
    print("Gridsearch complete!")
