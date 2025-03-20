import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import sys
import re
from collections import Counter

import initial_analyses as ia
sys.path.append("/local/disk2/not_backed_up/amukundan/research/vocal_disorder")
import query_mongo as query

MAIN_RCPD_SUBREDDITS = ["noburp"]
term_category_dict = {
    "Experience with RCPD": ['Faux pas', 'Anxiety', 'Social-anxiety', 'Flare-ups', 'Misdiagnosis', 'R-CPD', 'Isolation'], 
    "Symptoms": ['Vomit', 'Vomit air', 'Acid reflux', 'Air sick', 'Air burp', 'Air vomiting', 'Cough', 'Croaks', 'Chest pain', 'Gag', 'Gassy', 'Gurgling', 'Gurgles', 'Gas', 'Croaking', 'Internal burps', 'Pressure', 'Bloating', 'Retching', 'Reflux', 'Regurgitation', 'Symptoms', 'Shortness of breath', 'Throwing up', 'Throat tightness', 'Throat gurgles', 'Hiccups', 'Supragastric belching', 'Indigestion', 'Difficulty breathing', 'Gastrointestinal distress'],
    "Self-treatment methods": ['Chamomile', 'Tea', 'Exercises', 'Gas-X', 'Famotidine', 'Fizzy drinks', 'Omeprazole', 'Neck turning', 'Self-taught', 'Self-curing', 'Shaker', 'Pelvic tilt', 'Mounjaro', 'antacids', 'kiss the ceiling', 'Self-cured', 'Rapid Drink Challenge'],
    "Doctor based interventions": ['(Pre,post) -Botox', 'In-office procedure', 'surgery', 'Anesthesia', 'procedure', 'Units', 'Xeomin', 'Esophageal dilation', 'Injections', 'Saline'],
    "Associated possible conditions": ['hiatal hernia', 'Dyspepsia', 'GERD', 'Emetophobia', 'Abdomino phrenic dyssynergia (APD)', 'GI disorder', 'irritable bowel syndrome'], 
    "Who to seek for diagnosis/treatment": ['ENT', 'Gastroenterologist', 'Laryngologist', 'Otolaryngologist', 'PCP', 'Specialist', 'Insurance'], 
    "Diagnostic Steps": ['Endoscopy', 'Nasoendoscopy', 'Swallow tests', 'esophageal examination', 'HRM', 'Manometry', 'Fluoroscopy', 'Imaging', 'Barium swallow', 'FEES test'],
    "General Anatomy and Physiology Involved": ['Throat muscles', 'GI', 'Cricoid', 'Swallow', 'Peristalsis', 'Retrograde Cricopharyngeal Dysfunction']
}

raw_posts = query.get_posts_by_subreddits(MAIN_RCPD_SUBREDDITS)
processed_posts = [ia.preprocess_text(post['selftext']) for post in raw_posts]

unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

def tokenize(text):
    # Lowercase, then extract alphanumeric or apostrophe-containing tokens
    return re.findall(r"\b[\w']+\b", text.lower())

for doc in processed_posts:
    tokens = tokenize(doc)
    unigram_counter.update(tokens)
    
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + "_" + tokens[i+1]
        bigram_counter.update([bigram])
    
    for i in range(len(tokens) - 2):
        trigram = tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2]
        trigram_counter.update([trigram])

# Adjust frequency thresholds to capture enough terms without too much noise
min_freq_unigram = 5
min_freq_bigram  = 3
min_freq_trigram = 2

candidate_unigrams = [w for w, f in unigram_counter.items() if f >= min_freq_unigram]
candidate_bigrams  = [w for w, f in bigram_counter.items()  if f >= min_freq_bigram]
candidate_trigrams = [w for w, f in trigram_counter.items() if f >= min_freq_trigram]

candidate_terms = list(set(candidate_unigrams + candidate_bigrams + candidate_trigrams))
print(f"Unigrams (freq >= {min_freq_unigram}): {len(candidate_unigrams)}")
print(f"Bigrams (freq >= {min_freq_bigram}):  {len(candidate_bigrams)}")
print(f"Trigrams (freq >= {min_freq_trigram}): {len(candidate_trigrams)}")
print(f"Total candidate terms: {len(candidate_terms)}")


##############################################################################
# 3. LOAD PRETRAINED MODEL (BIO+CLINICALBERT) AND DEFINE EMBEDDING FUNCTION
##############################################################################

# model_name = "emilyalsentzer/Bio_ClinicalBERT"
model_name = "bioclinicalbert_noburp_pretrained/final_model"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained(model_name)
model.eval()  # inference mode

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, tokenizer, model, max_length=256, device="cpu"):
    """Embed a list of texts and return numpy arrays of shape (len(texts), hidden_dim)."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device)
        )
    embeddings = mean_pooling(outputs, encoded["attention_mask"].to(device))
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # normalization for better similarity
    return embeddings.cpu().numpy()


##############################################################################
# 4. EMBED EACH CATEGORY AND ITS SEED TERMS
##############################################################################

# Collect all unique seed terms from all categories to embed them in one batch
all_seed_terms = []
for cat, terms in term_category_dict.items():
    all_seed_terms.extend(terms)
unique_seed_terms = list(set(all_seed_terms))

# Embed seed terms
seed_embeddings = embed_texts(unique_seed_terms, tokenizer, model)

# Map seed term -> embedding
term2emb = {term: emb for term, emb in zip(unique_seed_terms, seed_embeddings)}

# Now compute average embedding per category
category_embeddings = {}
for cat, terms in term_category_dict.items():
    cat_vecs = []
    for t in terms:
        if t in term2emb:
            cat_vecs.append(term2emb[t])
        else:
            # If it's a multi-word term, we could embed it on the fly or handle token-splitting
            pass
    if cat_vecs:
        category_embeddings[cat] = np.mean(cat_vecs, axis=0)
    else:
        # fallback if no embeddings
        category_embeddings[cat] = np.zeros_like(seed_embeddings[0])

##############################################################################
# 5. EMBED YOUR CANDIDATE TERMS (UNIGRAMS, BIGRAMS, TRIGRAMS)
##############################################################################

# If you'd like to convert underscores back to spaces for better embedding, do so here:
candidate_texts_for_embedding = [term.replace("_", " ") for term in candidate_terms]

candidate_embeddings = {}
batch_size = 64
for i in range(0, len(candidate_texts_for_embedding), batch_size):
    batch = candidate_texts_for_embedding[i:i+batch_size]
    embs = embed_texts(batch, tokenizer, model)
    for original_term, embedded_term in zip(candidate_terms[i:i+batch_size], embs):
        candidate_embeddings[original_term] = embedded_term

##############################################################################
# 6. FIND MOST SIMILAR CANDIDATE TERMS FOR EACH CATEGORY
##############################################################################

expanded_terms = {}  # category -> list of (term, similarity)
for cat, cat_emb in category_embeddings.items():
    cat_emb_t = torch.tensor(cat_emb).unsqueeze(0)  # shape (1, hidden_dim)
    sim_list = []
    for cterm, cemb in candidate_embeddings.items():
        cemb_t = torch.tensor(cemb).unsqueeze(0)
        score = util.pytorch_cos_sim(cat_emb_t, cemb_t).item()  # cosine similarity
        sim_list.append((cterm, score))
    # Sort by descending similarity
    sim_list.sort(key=lambda x: x[1], reverse=True)
    
    # e.g. keep top 10 for demonstration
    top_n = 10
    expanded_terms[cat] = sim_list[:top_n]

##############################################################################
# 7. EXPAND EACH CATEGORY WITH THE NEW TERMS
##############################################################################

term_category_dict_expanded = {}
for cat in term_category_dict:
    # original terms
    original_terms_set = set(term_category_dict[cat])
    # new terms from expanded_terms
    top_matches = [t for t, s in expanded_terms[cat]]
    # merge
    updated_terms = list(original_terms_set.union(top_matches))
    term_category_dict_expanded[cat] = updated_terms

##############################################################################
# 8. SHOW WHAT TERMS GOT ADDED
##############################################################################

for cat in term_category_dict:
    original_set = set(term_category_dict[cat])
    updated_set  = set(term_category_dict_expanded[cat])
    newly_added  = updated_set - original_set
    
    print(f"CATEGORY: {cat}")
    print("New Terms Added:", newly_added)
    print("--------------------------------------------------")
