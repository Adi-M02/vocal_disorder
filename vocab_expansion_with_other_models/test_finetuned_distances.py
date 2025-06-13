from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import itertools
import json

# Load your fine-tuned ModernBERT model and tokenizer
finetuned_dir = "vocab_expansion_with_other_models/finetuned_modernBERT-base_06_11_13_48"
finetuned_dir = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
model = AutoModel.from_pretrained(finetuned_dir)

# Define your term lists (fill in with your actual lists)
# Load term lists from JSON file
with open('rcpd_terms_6_5.json', 'r') as f:
    term_lists = json.load(f)

# Function to compute average embedding of a list of terms
def average_list_embedding(terms):
    embeddings = []
    with torch.no_grad():
        for term in terms:
            inputs = tokenizer(term, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            # Mean-pool the token embeddings for the term
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings.append(emb)
    return torch.stack(embeddings).mean(dim=0)

# Compute average embeddings for each category
list_embeddings = {}
for name, terms in term_lists.items():
    list_embeddings[name] = average_list_embedding(terms)

# Compute pairwise cosine distances between all lists
names = list(list_embeddings.keys())
distances = {}
for a, b in itertools.combinations(names, 2):
    v1 = list_embeddings[a].unsqueeze(0)
    v2 = list_embeddings[b].unsqueeze(0)
    cos_sim = F.cosine_similarity(v1, v2).item()
    cos_dist = 1 - cos_sim
    distances[(a, b)] = cos_dist

# Display results
print("Pairwise cosine distances between term-list embeddings:")
for (a, b), dist in distances.items():
    print(f" - {a} vs {b}: {dist:.4f}")
