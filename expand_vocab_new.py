# vocab_expansion_flags.py
# Vocabulary expansion with support for maxsim, dynamic top-N, and KeyBERT

import torch
import numpy as np
import json
import re
import html
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download, data, pos_tag

tmp_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(tmp_dir, exist_ok=True)
nltk.data.path.append(tmp_dir)
for res in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]:
    try:
        data.find(f"corpora/{res}")
    except LookupError:
        download(res, download_dir="./nltk_data")

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

from rcpd_terms import rcpd_terms as TERM_CATEGORY_DICT
import query_mongo as query

# ─────────────────────────────────────────────────────────────────────────────

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in STOPWORDS)

def load_posts(subreddits):
    raw = query.get_posts_by_subreddits(subreddits, collection_name="noburp_all")
    def combine(p):
        return f"{p.get('title', '')} {p.get('selftext', '')}".strip() or p.get("body", "").strip()
    return [preprocess_text(combine(p)) for p in raw]

def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def generate_candidate_terms(posts):
    uni, bi, tri = Counter(), Counter(), Counter()
    for doc in posts:
        toks = tokenize(doc)
        uni.update(toks)
        bi.update([toks[i]+"_"+toks[i+1] for i in range(len(toks)-1)])
        tri.update(["_".join(toks[i:i+3]) for i in range(len(toks)-2)])
    return list(set([w for w in uni if uni[w]>=5] + [w for w in bi if bi[w]>=3] + [w for w in tri if tri[w]>=2]))

def embed_texts(texts, tokenizer, model, device, max_length=128, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move tensors to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling with proper device handling
        mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        mean = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # Normalize and move to CPU
        normed = torch.nn.functional.normalize(mean, p=2, dim=1)
        all_embeddings.append(normed.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

def embed_candidates(candidates, tokenizer, model, device):
    texts = [c.replace('_', ' ') for c in candidates]
    embeddings = embed_texts(texts, tokenizer, model, device, max_length=128, batch_size=32)
    return {c: emb for c, emb in zip(candidates, embeddings)}

def extract_keyphrases_per_post(posts, model_name, top_k=5, ngram_range=(1,3)):
    kw = KeyBERT(model_name)
    phrase_counts = Counter()
    for doc in posts:
        # limit doc length if needed: doc[:5000]
        phrases = kw.extract_keywords(
            doc,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_k
        )
        for p, _ in phrases:
            phrase_counts[p.lower()] += 1
    return phrase_counts

def filter_by_df(phrase_counts, min_df=3):
    return {phrase for phrase, cnt in phrase_counts.items() if cnt >= min_df}

def add_keybert_phrases(posts, term_dict, model_name, top_k=5, min_df=3):
    # 1) extract counts over all posts
    phrase_counts = extract_keyphrases_per_post(posts, model_name, top_k)
    # 2) threshold
    frequent = filter_by_df(phrase_counts, min_df)
    updated = {cat: set(terms) for cat, terms in term_dict.items()}
    for cat, seeds in term_dict.items():
        for phrase in frequent:
            # split phrase into words, compare stems or direct substring
            words = phrase.replace(" ", "_")
            if any(seed.lower() in phrase for seed in seeds):
                updated[cat].add(words)
    return {cat: list(terms) for cat, terms in updated.items()}

def expand_terms(term_dict, tokenizer, model, cand_embs, use_maxsim, dynamic_topn, top_n):
    seeds = list({t for terms in term_dict.values() for t in terms})
    term2emb = dict(zip(seeds, embed_texts([s.replace("_", " ") for s in seeds], tokenizer, model, device)))
    expanded = {}
    for cat, terms in term_dict.items():
        seed_vecs = [term2emb[t] for t in terms if t in term2emb]
        sims = {}
        for c, v in cand_embs.items():
            if use_maxsim:
                sims[c] = max(np.dot(v, s) for s in seed_vecs)
            else:
                sims[c] = np.dot(v, np.mean(seed_vecs, axis=0))
        sorted_terms = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        this_top_n = top_n  # for now; could scale by len(terms)
        expanded[cat] = list(set(terms) | set([c for c, _ in sorted_terms[:this_top_n]]))
    return expanded

if __name__ == "__main__":
    params = {
        "subreddits": ["noburp"],
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "top_n": 50,
        "use_maxsim": False,
        "dynamic_topn": False,
        "use_keybert": True,
        "keybert_model_name": "all-MiniLM-L6-v2",
    }

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    posts = load_posts(params["subreddits"])
    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    model = AutoModel.from_pretrained(params["model_name"]).to(device).eval()

    cands = generate_candidate_terms(posts)
    if params["use_keybert"]:
        print("🔍 Adding KeyBERT phrases...")
        term_dict = add_keybert_phrases(posts, TERM_CATEGORY_DICT, params["keybert_model_name"])
    else:
        term_dict = TERM_CATEGORY_DICT

    cand_embs = embed_candidates(cands, tokenizer, model, device)
    expanded = expand_terms(term_dict, tokenizer, model, cand_embs, params["use_maxsim"], params["dynamic_topn"], params["top_n"])

    output_payload = {
        "metadata": {
            **params,
            "generated_at": datetime.now(timezone.utc).isoformat()
        },
        "vocabulary": expanded
    }

    out_dir = Path(f"vocab_output_{datetime.now().strftime('%m_%d')}")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"expanded_output_{datetime.now().strftime('%m_%d_%H_%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved vocabulary to {out_path}")
