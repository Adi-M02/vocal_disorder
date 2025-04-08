import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
# Setup
model_name = "emilyalsentzer/Bio_ClinicalBERT"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# Embedding function
def embed_terms(terms, tokenizer, model, device="cpu"):
    with torch.no_grad():
        encoded = tokenizer(terms, padding=True, truncation=True, return_tensors="pt").to(device)
        model_output = model(**encoded)
        token_embeddings = model_output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = mask.sum(1)
        mean_pooled = summed / torch.clamp(counts, min=1e-9)
        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1).cpu().numpy()

# Terms
symptom_terms = [
    'Vomit air', 'Acid reflux', 'Air sick', 'Air burp', 'Air vomiting',
    'Croaks', 'Chest pain', 'Gassy', 'Gurgling', 'Gurgles', 'Croaking',
    'Internal burps', 'Bloating', 'Retching', 'Reflux', 'Regurgitation',
    'Shortness of breath', 'Throwing up', 'Throat tightness', 'Throat gurgles',
    'Hiccups', 'Supragastric belching', 'Indigestion', 'Difficulty breathing',
    'Gastrointestinal distress'
]
all_terms = symptom_terms + ['globus']

# Get embeddings
embeddings = embed_terms(all_terms, tokenizer, model, device=device)
term_to_embedding = {term: emb for term, emb in zip(all_terms, embeddings)}

# PCA plot
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
colors = ['blue'] * len(symptom_terms) + ['red']  # red for "globus"

plt.figure(figsize=(10, 6))
for i, term in enumerate(all_terms):
    plt.scatter(reduced[i, 0], reduced[i, 1], c=colors[i], s=100 if term == 'globus' else 60)
    plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, term, fontsize=9)
plt.title("PCA of Symptom Terms with 'globus'")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cosine similarity between proposed term and other terms in category
globus_vector = term_to_embedding['globus'].reshape(1, -1)
symptom_vectors = np.array([term_to_embedding[t] for t in symptom_terms])
similarities = cosine_similarity(globus_vector, symptom_vectors).flatten()

# Bar chart
x = np.arange(len(symptom_terms))
plt.figure(figsize=(12, 5))
bars = plt.bar(x, similarities, color='red')
for i, score in enumerate(similarities):
    plt.text(x[i], score + 0.01, f"{score:.2f}", ha='center', fontsize=9)

plt.xticks(x, symptom_terms, rotation=45, ha='right')
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity of 'globus' to Symptom Terms")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
