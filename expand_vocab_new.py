# vocab_expansion_flags.py
# Vocabulary expansion with support for maxsim, dynamic top-N, and KeyBERT,
# now using pretrained Bio-ClinicalBERT for all embedding.

import os
import sys
import re
import html
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download, data, pos_tag

from sentence_transformers import SentenceTransformer, models, util
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from multiprocessing import set_start_method

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK Setup
# ─────────────────────────────────────────────────────────────────────────────
tmp_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(tmp_dir, exist_ok=True)
nltk.data.path.append(tmp_dir)
for res in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]:
    try:
        data.find(f"corpora/{res}")
    except LookupError:
        download(res, download_dir=tmp_dir, quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────
with open('rcpd_terms.json', encoding="utf-8") as _f:
    rcpd_terms = json.load(_f)
import query_mongo as query

# ─────────────────────────────────────────────────────────────────────────────
# Text processing helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    return {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in STOPWORDS)

def load_posts(subreddits):
    raw = query.get_posts_by_subreddits(subreddits, collection_name="noburp_all")
    posts = []
    for p in raw:
        title   = p.get('title','').strip()
        selftext = p.get('selftext','') or p.get('body','')
        selftext = selftext.strip()
        # only add if non-empty
        if title or selftext:
            posts.append((
                preprocess_text(title)   if title   else None,
                preprocess_text(selftext) if selftext else None
            ))
    return posts

def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def generate_candidate_terms(posts):
    uni, bi, tri = Counter(), Counter(), Counter()
    for doc in posts:
        toks = tokenize(doc)
        uni.update(toks)
        bi.update([toks[i] + "_" + toks[i+1] for i in range(len(toks)-1)])
        tri.update(["_".join(toks[i:i+3]) for i in range(len(toks)-2)])
    return list(set(
        [w for w,f in uni.items() if f>=5] +
        [w for w,f in bi.items() if f>=3] +
        [w for w,f in tri.items() if f>=2]
    ))

def embed_texts(texts, tokenizer, model, device, max_length=128, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        mean = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        normed = torch.nn.functional.normalize(mean, p=2, dim=1)
        all_embs.append(normed.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

def embed_candidates(candidates, tokenizer, model, device):
    texts = [c.replace("_"," ") for c in candidates]
    embs = embed_texts(texts, tokenizer, model, device)
    return {c: e for c,e in zip(candidates, embs)}

# ─────────────────────────────────────────────────────────────────────────────
# KeyBERT + Bio-ClinicalBERT integration
# ─────────────────────────────────────────────────────────────────────────────
def build_bioclinical_keybert(model_path: str, device: str) -> KeyBERT:
    word_embed = models.Transformer(model_path, max_seq_length=512)
    pooling_layer = models.Pooling(
        word_embed.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    st = SentenceTransformer(modules=[word_embed, pooling_layer], device=device)
    return KeyBERT(model=st)

def extract_keyphrases_per_post_threaded(
    posts, kw_model, top_k=5, ngram_range=(1,3), max_workers=4
) -> Counter:
    phrase_counts = Counter()

    def _worker(pair):
        title, body = pair
        kws = []
        if title:
            kws += [p.lower() for p,_ in kw_model.extract_keywords(
                title, keyphrase_ngram_range=ngram_range,
                stop_words="english", top_n=top_k)]
        if body:
            kws += [p.lower() for p,_ in kw_model.extract_keywords(
                body, keyphrase_ngram_range=ngram_range,
                stop_words="english", top_n=top_k)]
        return kws

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_worker, doc) for doc in posts]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="KeyBERT", unit="doc"):
            phrase_counts.update(future.result())

    return phrase_counts

def filter_by_df(phrase_counts: Counter, min_df: int = 3) -> set:
    filtered = {p for p,c in phrase_counts.items() if c >= min_df}
    logger.info(f"[KeyBERT] Filtered to {len(filtered)} phrases with min_df ≥ {min_df}")
    return filtered

def add_keybert_phrases(
    posts,
    term_dict: dict,
    model_path: str,
    device: str,
    top_k: int = 5,
    min_df: int = 3,
    ngram_range: tuple = (1,3),
    max_workers: int = 4
) -> dict:
    logger.info(f"[KeyBERT] Building model at {model_path} on {device}")
    kw_model = build_bioclinical_keybert(model_path, device)

    logger.info(f"[KeyBERT] Starting phrase addition (top_k={top_k}, min_df={min_df})")
    counts  = extract_keyphrases_per_post_threaded(posts, kw_model, top_k, ngram_range, max_workers)
    frequent = filter_by_df(counts, min_df)

    updated = {cat:set(terms) for cat,terms in term_dict.items()}
    for cat, seeds in term_dict.items():
        before = len(updated[cat])
        for phrase in frequent:
            if any(seed.lower() in phrase for seed in seeds):
                updated[cat].add(phrase.replace(" ", "_"))
        after = len(updated[cat])
        logger.info(f"[KeyBERT] Category '{cat}': +{after-before} phrases (total {after})")

    logger.info(f"[KeyBERT] Merged into {len(term_dict)} categories")
    return {cat:list(terms) for cat, terms in updated.items()}


# ─────────────────────────────────────────────────────────────────────────────
# save keybert phrases before min_df filtering
# ─────────────────────────────────────────────────────────────────────────────

def run_keybert_extraction(
    posts,
    model_path: str,
    device: str,
    top_k: int = 5,
    ngram_range: tuple = (1,3),
    max_workers: int = 4,
) -> dict:
    """
    Runs KeyBERT up through building the 'phrase_counts' Counter.
    Returns a dict containing:
      - 'phrase_counts': plain dict mapping phrase -> doc frequency
      - 'metadata': dict of the parameters + run timestamp
    """
    # record run timestamp in MM_DD_HH_MM_SS format
    now = datetime.utcnow()
    run_ts = now.strftime('%m_%d_%H_%M_%S')

    # 1. Build model
    kw_model = build_bioclinical_keybert(model_path, device)

    # 2. Extract counts
    counts = extract_keyphrases_per_post_threaded(
        posts, kw_model,
        top_k=top_k,
        ngram_range=ngram_range,
        max_workers=max_workers
    )

    # 3. Bundle with metadata
    return {
        'metadata': {
            'run_timestamp': run_ts,
            'extraction_time': now.isoformat(),
            'model_path': model_path,
            'device': device,
            'top_k': top_k,
            'ngram_range': ngram_range,
            'max_workers': max_workers,
            'num_docs': len(posts),
        },
        'phrase_counts': dict(counts)  # Counter → plain dict
    }

def save_extraction_json(run_output: dict, directory: str = '.', prefix: str = 'keybert_run') -> str:
    """
    Serialize the run_output to a JSON file named:
      {prefix}_{MM_DD_HH_MM_SS}.json
    Returns the full path to the saved file.
    """
    ts = run_output['metadata']['run_timestamp']
    filename = f"{prefix}_{ts}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(run_output, f, ensure_ascii=False, indent=2)
    return filepath

def load_extraction_json(filepath: str):
    """
    Load a previously-saved JSON extraction.
    Returns:
      - phrase_counts: Counter
      - metadata: dict
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = Counter(data['phrase_counts'])
    metadata = data['metadata']
    return counts, metadata

# ─────────────────────────────────────────────────────────────────────────────
# assign keyphrases to categories
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_from_keybert(
    keyphrase_json_path: str,
    term_dict: dict,
    subreddits: list,
    model_name: str,
    device: str = "cpu",
    min_df: int = 3,
    min_sim: float = 0.6
) -> str:
    """
    1. Load phrase_counts & metadata from keyphrase_json_path
    2. Filter by min_df, embed all candidates
    3. Compute centroids on term_dict seeds
    4. Assign any candidate with cos_sim ≥ min_sim
    5. Save under vocab_output_MM_DD/expanded_output_MM_DD_HH_MM_SS.json
    Returns the output filepath.
    """
    # --- 1. load keyphrase run ---
    with open(keyphrase_json_path, 'r', encoding='utf-8') as f:
        run = json.load(f)
    phrase_counts = Counter(run['phrase_counts'])

    # --- 2. filter by min_df ---
    candidates = [p for p,c in phrase_counts.items() if c >= min_df]
    if not candidates:
        raise ValueError(f"No phrases survive min_df={min_df}")
    texts = [p.replace('_',' ') for p in candidates]

    # --- 3. embed candidates & seeds ---
    embed_model = SentenceTransformer(model_name, device=device)
    cand_embs = embed_model.encode(texts, device=device, convert_to_tensor=True)

    # compute category centroids
    centroids = {}
    for cat, seeds in term_dict.items():
        if not seeds:
            centroids[cat] = None
        else:
            # use seeds + category name as seed phrases
            emb_texts = [s.replace('_',' ') for s in seeds] + [cat.replace('_',' ')]
            seed_embs = embed_model.encode(emb_texts, device=device, convert_to_tensor=True)
            centroids[cat] = seed_embs.mean(dim=0)

    # --- 4. assign by fixed threshold ---
    updated = {cat:set(terms) for cat,terms in term_dict.items()}
    for cat, cent in centroids.items():
        if cent is None: continue
        sims = util.cos_sim(cent.unsqueeze(0), cand_embs)[0]
        hits = (sims >= min_sim).nonzero(as_tuple=True)[0].cpu().tolist()
        for idx in hits:
            phrase = candidates[idx].replace(' ', '_')
            updated[cat].add(phrase)

    # --- 5. prepare output JSON ---
    now = datetime.utcnow()
    mmdd      = now.strftime("%m_%d")
    timestamp = now.strftime("%m_%d_%H_%M_%S")
    out_dir = f"vocab_output_{mmdd}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"expanded_output_{timestamp}.json")

    out_json = {
        "metadata": {
            "generated_at": now.isoformat() + "+00:00",
            "subreddits": subreddits,
            "candidate_generation_method": "keybert_threshold",
            "source_keyphrase_json": keyphrase_json_path,
            "model_name": model_name,
            "min_df": min_df,
            "min_sim": min_sim
        },
        "vocabulary": {
            cat: sorted(list(terms))
            for cat, terms in updated.items()
        }
    }

    # --- 6. save ---
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# Term-expansion pipeline
# ─────────────────────────────────────────────────────────────────────────────
def expand_terms(term_dict, tokenizer, model, cand_embs,
                 use_maxsim, dynamic_topn, top_n, device):
    seeds = list({t for terms in term_dict.values() for t in terms})
    seed_embs = embed_texts([s.replace("_"," ") for s in seeds], tokenizer, model, device)
    term2emb = dict(zip(seeds, seed_embs))

    expanded = {}
    for cat, terms in term_dict.items():
        seed_vecs = [term2emb[t] for t in terms if t in term2emb]
        sims = {}
        for c,v in cand_embs.items():
            sims[c] = (max(np.dot(v,s) for s in seed_vecs)
                       if use_maxsim else
                       np.dot(v, np.mean(seed_vecs, axis=0)))
        sorted_terms = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        this_top_n = top_n
        expanded[cat] = list(set(terms) | {c for c,_ in sorted_terms[:this_top_n]})
    return expanded

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_start_method("spawn", force=True)
    
    params = {
        "subreddits": ["noburp"],
        # use your continued-pretrained directory here
        "finetuned_dir": "emilyalsentzer/Bio_ClinicalBERT",
        "top_n": 50,
        "use_maxsim": False,
        "dynamic_topn": False,
        "use_keybert": False,
        "min_df": 3,
        "max_workers": 8,
    }

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load and preprocess posts
    # posts = load_posts(params["subreddits"])

    #save keybert phrases before min_df filtering

    # model_path  = "emilyalsentzer/Bio_ClinicalBERT"
    # device      = "cuda"
    # max_workers = 4
    # prefix      = "keybert_run"

    # for top_k in range(1, 11):                  # 1 → 10
    #     for ngram_range in [(1,3), (2,4)]:     # (1,3) and (2,4)
    #         print(f"▶ Running top_k={top_k}, ngram_range={ngram_range}")
    #         run_output = run_keybert_extraction(
    #             posts,
    #             model_path=model_path,
    #             device=device,
    #             top_k=top_k,
    #             ngram_range=ngram_range,
    #             max_workers=max_workers,
    #         )
    #         # embed parameters in filename
    #         run_prefix = f"{prefix}_k{top_k}_ng{ngram_range[0]}-{ngram_range[1]}"
    #         saved_path = save_extraction_json(
    #             run_output,
    #             directory="keybert_outputs",
    #             prefix=run_prefix
    #         )
    #         print(f"✔ Saved to {saved_path}\n")
    # sys.exit(0)

    # load keybert phrases and run min_df filtering and asssign to categories
    vocab_file = build_vocab_from_keybert(
        keyphrase_json_path="keybert_outputs/keybert_run_k4_ng2-4_05_08_13_11_52.json",
        term_dict=TERM_CATEGORY_DICT,
        subreddits=["noburp"],
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        device="cuda",
        min_df=1,
        min_sim=0.92
    )
    print("Saved vocabulary to:", vocab_file)

    # # Load pretrained tokenizer & model from local dir
    # # tokenizer = AutoTokenizer.from_pretrained(os.path.join(params["finetuned_dir"], "tokenizer"))
    # # model     = AutoModel.from_pretrained(params["finetuned_dir"]).to(device).eval()
    # # load off the shelf Bio-ClinicalBERT
    # tokenizer = AutoTokenizer.from_pretrained(params["finetuned_dir"])
    # model = AutoModel.from_pretrained(params["finetuned_dir"]).to(device)

    # # Generate candidate terms
    # cands = generate_candidate_terms(posts)

    # # KeyBERT expansion with continued-pretrained Bio-ClinicalBERT
    # if params["use_keybert"]:
    #     term_dict = add_keybert_phrases(
    #         posts,
    #         TERM_CATEGORY_DICT,
    #         params["finetuned_dir"],
    #         device,
    #         top_k=params["top_n"],
    #         min_df=params["min_df"],
    #         ngram_range=(1,3),
    #         max_workers=params["max_workers"]
    #     )
    # else:
    #     term_dict = TERM_CATEGORY_DICT

    # # Embed candidates & expand with your pretrained model
    # if params["use_keybert"]:
    #     expanded = term_dict
        
    # else:
    #     cand_embs = embed_candidates(cands, tokenizer, model, device)
    #     expanded = expand_terms(
    #         term_dict,
    #         tokenizer,
    #         model,
    #         cand_embs,
    #         params["use_maxsim"],
    #         params["dynamic_topn"],
    #         params["top_n"],
    #         device
    #     )

    # # Save output
    # output = {
    #     "metadata": {**params, "generated_at": datetime.now(timezone.utc).isoformat()},
    #     "vocabulary": expanded
    # }
    # out_dir = Path(f"vocab_output_{datetime.now().strftime('%m_%d')}")
    # out_dir.mkdir(exist_ok=True)
    # out_path = out_dir / f"expanded_output_{datetime.now().strftime('%m_%d_%H_%M')}.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)
    # logger.info(f"✅ Saved vocabulary to {out_path}")
