import os
import sys
import json
import math
import argparse
import itertools
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from query_mongo import return_documents

def embed_phrase_bert(
    phrase: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device
) -> Optional[np.ndarray]:
    """Tokenize phrase, run through BERT, mean-pool last hidden state."""
    inputs = tokenizer(
        phrase,
        return_tensors="pt",
        truncation=True,
        max_length=32,
        add_special_tokens=True
    ).to(device)

    with torch.no_grad():
        out = model(**inputs).last_hidden_state  # (1, seq_len, hidden_size)

    # mask out padding tokens when averaging
    attention_mask = inputs.attention_mask.unsqueeze(-1)  # (1, seq_len, 1)
    summed = (out * attention_mask).sum(dim=1)            # (1, hidden_size)
    counts = attention_mask.sum(dim=1).clamp(min=1)       # (1, 1)
    return (summed / counts).squeeze(0).cpu().numpy()     # (hidden_size,)


def load_terms(path: str) -> dict[str, list[str]]:
    """Load category→list-of-terms from JSON, replacing underscores with spaces."""
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }


def precompute_bert_term_vectors(
    db_name: str,
    collection_name: str,
    tokenizer,                    # AutoTokenizer already loaded
    model,                        # AutoModel already loaded
    device: torch.device,         # e.g. torch.device("cuda"/"cpu")
    min_count: int = 5,
    mongo_uri: str = "mongodb://localhost:27017/",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    1) Fetch all docs
    2) Tokenize, count unigrams & adjacent bigrams
    3) Filter out any n-gram with a stopword or count < min_count
    4) Embed each surviving unigram & bigram with BERT (using the passed-in model/tokenizer)
    Returns two dicts: term_to_vec, bigram_to_vec
    """
    # 1) fetch and tokenize everything
    docs = return_documents(db_name, collection_name, filter_subreddits=["noburp"], mongo_uri=mongo_uri)
    unigram_counts = Counter()
    bigram_counts  = Counter()
    stop_words = set(stopwords.words("english"))

    for doc_text in docs:
        tokens = clean_and_tokenize(doc_text)
        unigram_counts.update(tokens)
        bigram_counts.update(zip(tokens, tokens[1:]))

    # 2) filter by count & stop-words
    unigrams = [
        w for w, cnt in unigram_counts.items()
        if cnt >= min_count and w not in stop_words
    ]
    bigrams = [
        (w1, w2) for (w1, w2), cnt in bigram_counts.items()
        if cnt >= min_count and w1 not in stop_words and w2 not in stop_words
    ]
    bigram_phrases = [" ".join(bg) for bg in bigrams]

    # 3) embed & store using the provided tokenizer/model
    term_to_vec   : Dict[str, np.ndarray] = {}
    bigram_to_vec : Dict[str, np.ndarray] = {}

    for w in unigrams:
        vec = embed_phrase_bert(w, tokenizer, model, device)
        term_to_vec[w] = vec

    for phrase in bigram_phrases:
        vec = embed_phrase_bert(phrase, tokenizer, model, device)
        bigram_to_vec[phrase] = vec

    print(f"Precomputed {len(term_to_vec)} unigram vectors and "
          f"{len(bigram_to_vec)} bigram vectors.")
    return term_to_vec, bigram_to_vec


def compute_triplet_vectors(
    bert_model,                     # your AutoModel
    tokenizer,                      # your AutoTokenizer
    device,                         # torch.device("cuda"/"cpu")
    terms_map: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    For each category, take all 2-term combinations of its seeds.
    Embed term1, term2, and category name with BERT, then average those three vectors.
    """
    triplets: List[Dict[str, Any]] = []

    for category, seeds in terms_map.items():
        # embed the category name
        cat_vec: Optional[np.ndarray] = embed_phrase_bert(
            category, tokenizer, bert_model, device
        )
        if cat_vec is None:
            print(f"Warning: Category name '{category}' OOV; skipping")
            continue

        # for each pair of seed terms
        for t1, t2 in itertools.combinations(seeds, 2):
            v1 = embed_phrase_bert(t1, tokenizer, bert_model, device)
            v2 = embed_phrase_bert(t2, tokenizer, bert_model, device)
            if v1 is None or v2 is None:
                continue

            # average the three vectors
            trip_vector = (v1 + v2 + cat_vec) / 3.0
            triplets.append({
                "category": category,
                "vector": trip_vector
            })

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Expand categories using triplet + frequency filtering"
    )
    parser.add_argument('--terms',       required=True, help='Path to rcpd_terms.json')
    parser.add_argument('--sim_threshold', type=float, default=0.4,
                        help='Cosine‐sim cutoff (default 0.4)')
    parser.add_argument('--freq_threshold', type=float, default=0.09,
                        help='Min fraction of triplets a term must appear in (default 0.09)')
    args = parser.parse_args()

    # 1) load seed terms
    terms_map = load_terms(args.terms)
    print(f"Loaded {sum(len(v) for v in terms_map.values())} terms "
          f"across {len(terms_map)} categories")

    MODELS = [
    ("bert-base",      "bert-base-uncased"),
    ("bertweet",       "vinai/bertweet-base"),
    ("bioclinical-bert",  "emilyalsentzer/Bio_ClinicalBERT"),
    ("pubmed-bert",    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # make dir for expansions for all models each run
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    model_dir = os.path.join(BASE_DIR, "expansions", timestamp)
    os.makedirs(model_dir, exist_ok=True)

    for name, model_filename in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_filename, use_fast=True)
        model     = AutoModel.from_pretrained(model_filename).to(device)
        model.eval()

        term_to_vec, bigram_to_vec = precompute_bert_term_vectors(
            db_name="reddit",
            collection_name="noburp_all",
            min_count=5,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        # 2) turn them into the “vocab matrix” + norms
        vocab_words  = list(term_to_vec.keys())
        vocab_matrix = np.vstack([term_to_vec[w] for w in vocab_words])
        vocab_norms  = np.linalg.norm(vocab_matrix, axis=1)

        bigram_phrases = list(bigram_to_vec.keys())
        bigram_matrix  = np.vstack([bigram_to_vec[b] for b in bigram_phrases])
        bigram_norms   = np.linalg.norm(bigram_matrix, axis=1)

        # compute triplet vectors
        triplets = compute_triplet_vectors(model, tokenizer, device, terms_map)
        print(f"Computed {len(triplets)} triplet vectors")

        # collect per-triplet candidate sets
        per_triplet: dict[str, list[set[str]]] = {cat: [] for cat in terms_map}
        for trip in triplets:
            cat = trip['category']
            vec = trip['vector']
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue

            # 1) unigrams
            sims = vocab_matrix.dot(vec) / (vocab_norms * norm)
            idxs = np.where(sims >= args.sim_threshold)[0]
            candidates = {vocab_words[i] for i in idxs}

            # 2) bigrams
            sims_bi = bigram_matrix.dot(vec) / (bigram_norms * norm)
            bi_idxs = np.where(sims_bi >= args.sim_threshold)[0]
            for j in bi_idxs:
                candidates.add(bigram_phrases[j])

            per_triplet[cat].append(candidates)

        # 4) frequency filtering
        expansions = {}
        for cat, subsets in per_triplet.items():
            total_queries = len(subsets)
            if total_queries == 0:
                expansions[cat] = []
                continue
            # Minimum times a term must appear
            min_count = math.ceil(args.freq_threshold * total_queries)
            # Count across all triplet‐sets
            counter = Counter()
            for s in subsets:
                counter.update(s)
            # Keep those meeting both sim + freq criteria
            kept = [term for term, cnt in counter.items() if cnt >= min_count]
            expansions[cat] = kept
            print(f"{model_filename} | {cat}: "
                  f"{len(terms_map[cat])} seeds → +{len(kept)} expansions "
                  f"(freq ≥ {min_count}/{total_queries})")

        # 5) merge seeds + expansions & write JSON
        merged = {
            cat: terms_map[cat] + expansions[cat]
            for cat in terms_map
        }
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        
        out_file = os.path.join(
            model_dir,
            f"{name}_expansion.json"
        )
        with open(out_file, 'w', encoding='utf-8') as fw:
            json.dump(merged, fw, indent=2)
        print(f"Wrote expansion to {out_file}")

    print("\nDone.")


if __name__ == '__main__':
    main()
