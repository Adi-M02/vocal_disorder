#!/usr/bin/env python3
import os
import re
import html
import json
import torch
import nltk
import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("evaluate_vocab", 'vocabulary_evaluation/evaluate_vocab.py')
evaluate_vocab_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_vocab_module)

# OPTIONAL LSA ------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
# -------------------------------------------------

import query_mongo as query
from rcpd_terms import rcpd_terms as TERM_CATEGORY_DICT

# =========== NLTK setup ===========
tmp_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(tmp_dir, exist_ok=True)
nltk.data.path.append(tmp_dir)
for res in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, download_dir=tmp_dir, quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# =========== Globals & helpers ===========
MAIN_RCPD_SUBREDDITS = ["noburp"]
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

# --------- Text preprocessing ---------

def preprocess_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    tokens = [w for w in word_tokenize(text) if w not in STOPWORDS]
    return " ".join(tokens)

def tokenize(text: str):
    return re.findall(r"\b[\w']+\b", text.lower())

# --------- Candidate generation ---------

def generate_ngram_candidates(posts, min_uni=5, min_bi=3, min_tri=2):
    uni, bi, tri = Counter(), Counter(), Counter()
    for doc in posts:
        toks = tokenize(doc)
        uni.update(toks)
        for i in range(len(toks) - 1):
            bi.update([f"{toks[i]}_{toks[i+1]}"])
        for i in range(len(toks) - 2):
            tri.update([f"{toks[i]}_{toks[i+1]}_{toks[i+2]}"])
    u = [w for w, f in uni.items() if f >= min_uni]
    b = [w for w, f in bi.items() if f >= min_bi]
    t = [w for w, f in tri.items() if f >= min_tri]
    return set(u + b + t)

# --------- OPTIONAL: rule-based morphology ---------

def morphological_variants(term: str):
    term = term.lower()
    if " " not in term:
        base = lemmatizer.lemmatize(term)
        return {term, base + "s", base + "ing", base + "ed"}
    parts = term.split()
    head, tail = " ".join(parts[:-1]), parts[-1]
    base_tail = lemmatizer.lemmatize(tail)
    vars_tail = {tail, base_tail + "s", base_tail + "ing", base_tail + "ed"}
    return {f"{head} {v}".strip() for v in vars_tail}

# --------- BERT embedding helpers ---------

def mean_pooling(model_out, mask):
    token_embs = model_out.last_hidden_state
    expanded = mask.unsqueeze(-1).expand(token_embs.size()).float()
    return (token_embs * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)

def embed_texts(texts, tokenizer, model, device="cpu", max_len=256):
    if not texts:
        return np.empty((0, 768))
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    vecs = mean_pooling(out, enc["attention_mask"].to(device))
    return torch.nn.functional.normalize(vecs, p=2, dim=1).cpu().numpy()

# ─────────────── Batched wrapper ───────────────
def embed_texts_batched(texts, tokenizer, model, device="cpu", max_len=256, batch_size=64):
    """Split `texts` into batches before embedding to avoid OOM."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_texts(batch, tokenizer, model, device, max_len)
        all_vecs.append(vecs)
    if all_vecs:
        return np.vstack(all_vecs)
    else:
        return np.empty((0, 768))

# --------- OPTIONAL: LSA builders ---------

def build_lsa_embeddings(docs, vocab, n_components=250, min_df=2):
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for LSA but not found.")
    vect = TfidfVectorizer(vocabulary=list(vocab), ngram_range=(1, 3), min_df=min_df, lowercase=False)
    X = vect.fit_transform(docs)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Z = svd.fit_transform(X.T)
    Z = normalize(Z)
    term_order = vect.get_feature_names_out()
    return {t: Z[i] for i, t in enumerate(term_order)}

# --------- Expansion strategies ---------

def expand_bert(term_dict, cand2emb, top_n):
    seeds_flat = {t for ts in term_dict.values() for t in ts}
    term2emb = {**cand2emb, **{s: cand2emb.get(s) for s in seeds_flat if s in cand2emb}}
    centroids = {
        cat: np.mean([term2emb[t] for t in ts if t in term2emb], axis=0)
        for cat, ts in term_dict.items()
    }
    out = {}
    for cat, cent in centroids.items():
        sims = [(t, float(np.dot(v, cent))) for t, v in cand2emb.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        out[cat] = [t for t, _ in sims[:top_n]]
    return {c: sorted(set(term_dict[c]) | set(out[c])) for c in term_dict}

def expand_lsa(term_dict, cand2vec, top_n):
    seeds_flat = {t for ts in term_dict.values() for t in ts if t in cand2vec}
    centroids = {c: np.mean([cand2vec[t] for t in ts if t in cand2vec], axis=0) for c, ts in term_dict.items()}
    out = {}
    for cat, cent in centroids.items():
        sims = [(t, float(np.dot(v, cent))) for t, v in cand2vec.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        out[cat] = [t for t, _ in sims[:top_n]]
    return {c: sorted(set(term_dict[c]) | set(out[c])) for c in term_dict}

def expand_hybrid(term_dict, cand2bert, cand2lsa, seeds_bert, seeds_lsa, alpha=0.5, top_n=60):
    out = {}
    for cat, seeds in term_dict.items():
        bert_cent = np.mean([seeds_bert[s] for s in seeds if s in seeds_bert], axis=0)
        lsa_cent = np.mean([seeds_lsa[s] for s in seeds if s in seeds_lsa], axis=0) if seeds_lsa else None
        scores = []
        for t in cand2bert.keys():
            sim_b = float(np.dot(cand2bert[t], bert_cent))
            sim = sim_b
            if lsa_cent is not None and t in cand2lsa:
                sim_l = float(np.dot(cand2lsa[t], lsa_cent))
                sim = alpha * sim_b + (1 - alpha) * sim_l
            scores.append((t, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        out[cat] = [t for t, _ in scores[:top_n]]
    return {c: sorted(set(term_dict[c]) | set(out[c])) for c in term_dict}

# --------- Main pipeline ---------

def expand_category_terms_pipeline(
    subreddits=MAIN_RCPD_SUBREDDITS,
    min_freq_unigram=5,
    min_freq_bigram=3,
    min_freq_trigram=2,
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    batch_size=64,
    top_n=60,
    device="cpu",
    term_category_dict=None,
    enable_morphology=False,
    enable_lsa=False,
    enable_hybrid=False,
    alpha=0.5,
):
    if enable_lsa and not SKLEARN_AVAILABLE:
        raise RuntimeError("enable_lsa=True but scikit-learn not installed.")
    if enable_hybrid and not enable_lsa:
        raise ValueError("Hybrid requires enable_lsa=True.")
    if term_category_dict is None:
        term_category_dict = TERM_CATEGORY_DICT

    raw_posts = query.get_posts_by_subreddits(subreddits, collection_name="noburp_all")
    docs = [preprocess_text(f"{p.get('title','')} {p.get('selftext','')}") for p in raw_posts]

    cands = generate_ngram_candidates(docs, min_freq_unigram, min_freq_bigram, min_freq_trigram)
    if enable_morphology:
        seeds = {t for ts in term_category_dict.values() for t in ts}
        for s in seeds:
            cands.update(morphological_variants(s))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # — embed candidates in batches
    cand_list = list(cands)
    cand_texts = [t.replace("_", " ") for t in cand_list]
    cand_vecs  = embed_texts_batched(cand_texts, tokenizer, model, device, max_len=256, batch_size=batch_size)
    cand2bert  = dict(zip(cand_list, cand_vecs))

    # — embed seeds in batches
    seeds_flat = list({t for ts in term_category_dict.values() for t in ts})
    seed_texts = [t.replace("_", " ") for t in seeds_flat]
    seed_vecs  = embed_texts_batched(seed_texts, tokenizer, model, device, max_len=256, batch_size=batch_size)
    seeds_bert = dict(zip(seeds_flat, seed_vecs))

    cand2lsa = seeds_lsa = None
    if enable_lsa:
        cand2lsa = build_lsa_embeddings(docs, cands, n_components=250, min_df=2)
        seeds_lsa = {t: cand2lsa[t] for t in seeds_flat if t in cand2lsa}

    if enable_hybrid:
        expanded = expand_hybrid(term_category_dict, cand2bert, cand2lsa, seeds_bert, seeds_lsa, alpha, top_n)
    elif enable_lsa:
        expanded = expand_lsa(term_category_dict, cand2lsa, top_n)
    else:
        expanded = expand_bert(term_category_dict, cand2bert, top_n)

    return expanded, {
        "enable_morphology": enable_morphology,
        "enable_lsa": enable_lsa,
        "enable_hybrid": enable_hybrid,
        "alpha": alpha if enable_hybrid else None,
    }

# ===================== CLI & save =====================
if __name__ == "__main__":
    params = {
        "subreddits": MAIN_RCPD_SUBREDDITS,
        "min_freq_unigram": 3,
        "min_freq_bigram": 2,
        "min_freq_trigram": 1,
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "batch_size": 64,
        "top_n": 60,
        "enable_morphology": False,
        "enable_lsa": True,
        "enable_hybrid": True,
        "alpha": 0.5,
    }
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    vocab_dict, flag_meta = expand_category_terms_pipeline(
        subreddits=params["subreddits"],
        min_freq_unigram=params["min_freq_unigram"],
        min_freq_bigram=params["min_freq_bigram"],
        min_freq_trigram=params["min_freq_trigram"],
        model_name=params["model_name"],
        batch_size=params["batch_size"],
        top_n=params["top_n"],
        device=device,
        enable_morphology=params["enable_morphology"],
        enable_lsa=params["enable_lsa"],
        enable_hybrid=params["enable_hybrid"],
        alpha=params["alpha"],
        term_category_dict=TERM_CATEGORY_DICT,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subreddits": params["subreddits"],
        "model_name": params["model_name"].split("/")[-1],
        "top_n": params["top_n"],
        "min_freqs": {
            "uni": params["min_freq_unigram"],
            "bi": params["min_freq_bigram"],
            "tri": params["min_freq_trigram"],
        },
        **flag_meta,
    }
    payload = {"metadata": metadata, "vocabulary": vocab_dict}

    Path("vocab_output").mkdir(exist_ok=True)
    file_name = f"expanded_vocab_{datetime.now().strftime('%m_%d_%H%M')}.json"
    out_path = Path("vocab_output") / file_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved to {out_path}")
    evaluate_vocab_module.evaluate_vocab(out_path, 'vocabulary_evaluation/manual_terms.txt', usernames = ["freddiethecalathea", "Many_Pomegranate_566", "rpesce518", "kinglgw", "mjh59"], )
