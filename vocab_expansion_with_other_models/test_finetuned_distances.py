from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import itertools
import json

# Load your fine-tuned ModernBERT model and tokenizer
finetuned_dir = "vocab_expansion_with_other_models/finetuned_modernBERT-base_06_11_13_48/"
tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
model = AutoModel.from_pretrained(finetuned_dir)

# Load term lists from JSON file
with open('rcpd_terms_6_5.json', 'r') as f:
    term_lists = json.load(f)

# Function to compute average embedding of a list of terms, each prefixed by category name
def average_list_embedding(category, terms):
    embeddings = []
    with torch.no_grad():
        for term in terms:
            text = f"{category} {term}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            # Mean-pool the token embeddings for the text
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings.append(emb)
    return torch.stack(embeddings).mean(dim=0)

# Compute average embeddings for each category using category+term context
list_embeddings = {}
for name, terms in term_lists.items():
    list_embeddings[name] = average_list_embedding(name, terms)

import faiss
import numpy as np

# Prepare embeddings matrix
names = list(list_embeddings.keys())
emb_matrix = torch.stack([list_embeddings[name] for name in names]).cpu().numpy()

# Normalize embeddings to unit length for cosine similarity
def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

emb_norm = l2_normalize(emb_matrix)

# Build FAISS GPU index for inner product search (cosine similarity after normalization)
dim = emb_norm.shape[1]
res = faiss.StandardGpuResources()             # use a single GPU
index = faiss.GpuIndexFlatIP(res, dim)         # inner product on GPU
index.add(emb_norm)

# Search against all embeddings
D, I = index.search(emb_norm, len(names))      # distances and indices for all pairs

distances = {}
for i in range(len(names)):
    for idx_j, j in enumerate(I[i]):
        if j <= i:
            continue  # skip self and duplicates
        sim = D[i][idx_j]
        cos_dist = 1 - sim
        distances[(names[i], names[j])] = cos_dist

# Display results
print("Pairwise cosine distances between category-term embeddings (via FAISS GPU):")
for (a, b), dist in distances.items():
    print(f" - {a} vs {b}: {dist:.4f}")
