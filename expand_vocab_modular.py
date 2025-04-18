import torch
import numpy as np
import json
from sentence_transformers import util
import re
from collections import Counter
import query_mongo as query
import html
import os
import nltk
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Set up NLTK environment
tmp_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(tmp_dir, exist_ok=True)
nltk.data.path.append(tmp_dir)
for resource in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=tmp_dir, quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from rcpd_terms import rcpd_terms as TERM_CATEGORY_DICT

MAIN_RCPD_SUBREDDITS = ["noburp"]
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    from nltk import pos_tag
    tag = pos_tag([word])[0][1][0].upper()
    return {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in STOPWORDS)

def preprocess_text_lemmatize(text):
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    words = word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words if w not in STOPWORDS)

def load_and_preprocess_posts(subreddits, query_module, preprocess_fn):
    raw = query_module.get_posts_by_subreddits(subreddits, collection_name="noburp_posts")
    return [preprocess_fn(p.get('selftext', '')) for p in raw]

def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def generate_candidate_terms(posts, min_freq_unigram=5, min_freq_bigram=3, min_freq_trigram=2):
    uni, bi, tri = Counter(), Counter(), Counter()
    for doc in posts:
        toks = tokenize(doc)
        uni.update(toks)
        for i in range(len(toks) - 1): bi.update([toks[i] + '_' + toks[i+1]])
        for i in range(len(toks) - 2): tri.update([toks[i] + '_' + toks[i+1] + '_' + toks[i+2]])
    u = [w for w, f in uni.items() if f >= min_freq_unigram]
    b = [w for w, f in bi.items() if f >= min_freq_bigram]
    t = [w for w, f in tri.items() if f >= min_freq_trigram]
    print(f"Candidates (ngram): {len(set(u + b + t))}")
    return list(set(u + b + t))

def generate_candidate_terms_wordnet(term_dict):
    syns = set()
    for terms in term_dict.values():
        for term in terms:
            for s in wordnet.synsets(term):
                for l in s.lemma_names(): syns.add(l.lower())
    print(f"Candidates (wordnet): {len(syns)}")
    return list(syns)

def extract_ngrams(posts, max_n=3, min_freq=1):
    ctr = Counter()
    for doc in posts:
        toks = tokenize(doc)
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1): ctr['_'.join(toks[i:i+n])] += 1
    return [ng for ng, f in ctr.items() if f >= min_freq]

def embed_candidates(candidates, tokenizer, model, device="cpu", batch_size=64):
    texts = [c.replace('_', ' ') for c in candidates]
    embs = {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        arr = embed_texts(batch, tokenizer, model, device)
        for term, v in zip(candidates[i:i + batch_size], arr):
            embs[term] = v
    return embs

def mean_pooling(model_out, mask):
    toks = model_out.last_hidden_state
    m = mask.unsqueeze(-1).expand(toks.size()).float()
    return (toks * m).sum(1) / m.sum(1).clamp(min=1e-9)

def embed_texts(texts, tokenizer, model, device="cpu", max_length=256):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))
    embs = mean_pooling(out, encoded['attention_mask'].to(device))
    return torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy()

def expand_terms_category_dict(term_dict, tokenizer, model, cand_embs, top_n=10, device="cpu"):
    seeds = list({t for terms in term_dict.values() for t in terms})
    seed_embs = embed_texts([s.replace('_', ' ') for s in seeds], tokenizer, model, device)
    term2emb = dict(zip(seeds, seed_embs))
    centroids = {
        cat: np.mean([term2emb[t] for t in terms if t in term2emb], axis=0)
        if any(t in term2emb for t in terms) else np.zeros_like(next(iter(cand_embs.values())))
        for cat, terms in term_dict.items()
    }
    expanded = {}
    for cat, cent in centroids.items():
        sim = [(c, util.pytorch_cos_sim(torch.tensor(cent).unsqueeze(0), torch.tensor(v).unsqueeze(0)).item()) for c, v in cand_embs.items()]
        sim.sort(key=lambda x: x[1], reverse=True)
        expanded[cat] = [c for c, _ in sim[:top_n]]
    return {cat: list(set(term_dict.get(cat, [])) | set(expanded.get(cat, []))) for cat in term_dict}

def expand_category_terms_pipeline(subreddits, query_module, preprocess_fn,
                                   candidate_generation_method="ngram",
                                   min_freq_unigram=5, min_freq_bigram=3, min_freq_trigram=2,
                                   model_name="emilyalsentzer/Bio_ClinicalBERT",
                                   batch_size=64, term_category_dict=None,
                                   top_n=10, device="cuda"):
    posts = load_and_preprocess_posts(subreddits, query_module, preprocess_fn)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if candidate_generation_method == "ngram":
        cands = generate_candidate_terms(posts, min_freq_unigram, min_freq_bigram, min_freq_trigram)
    elif candidate_generation_method == "wordnet":
        cands = generate_candidate_terms_wordnet(term_category_dict)
    elif candidate_generation_method == "embedding":
        raw = extract_ngrams(posts, max_n=3, min_freq=min_freq_unigram)
        emb_cands = embed_candidates(raw, tokenizer, model, device, batch_size)
        return expand_terms_category_dict(term_category_dict, tokenizer, model, emb_cands, top_n, device)
    elif candidate_generation_method == "combined":
        uni = set(generate_candidate_terms(posts, min_freq_unigram, min_freq_bigram, min_freq_trigram))
        wnet = set(generate_candidate_terms_wordnet(term_category_dict))
        cands = list(uni.union(wnet))
    else:
        raise ValueError(f"Unknown method: {candidate_generation_method}")

    emb_cands = embed_candidates(cands, tokenizer, model, device, batch_size)
    return expand_terms_category_dict(term_category_dict, tokenizer, model, emb_cands, top_n, device)

if __name__ == "__main__":
    params = {
        "subreddits": MAIN_RCPD_SUBREDDITS,
        "candidate_generation_method": "combined",
        "min_freq_unigram": 5,
        "min_freq_bigram": 3,
        "min_freq_trigram": 2,
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "batch_size": 64,
        "top_n": 20,
    }

    updated_dict = expand_category_terms_pipeline(
        subreddits=params["subreddits"],
        query_module=query,
        preprocess_fn=preprocess_text,
        candidate_generation_method=params["candidate_generation_method"],
        min_freq_unigram=params["min_freq_unigram"],
        min_freq_bigram=params["min_freq_bigram"],
        min_freq_trigram=params["min_freq_trigram"],
        model_name=params["model_name"],
        batch_size=params["batch_size"],
        term_category_dict=TERM_CATEGORY_DICT,
        top_n=params["top_n"],
        device="cuda"
    )

    output_payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **params
        },
        "vocabulary": updated_dict
    }

    with open("expanded_vocab_with_metadata.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print("✅ Saved expanded_vocab_with_metadata.json")

    folder_date = datetime.now().strftime("%m_%d")
    output_folder = Path(f"vocab_output_{folder_date}")
    output_folder.mkdir(parents=True, exist_ok=True)

    run_tag = (
        f"{params['candidate_generation_method']}"
        f"_top{params['top_n']}"
        f"_uf{params['min_freq_unigram']}"
        f"_bf{params['min_freq_bigram']}"
        f"_tf{params['min_freq_trigram']}"
    )

    output_path = output_folder / f"expanded_vocab_{run_tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Vocabulary + metadata saved to {output_path}")
