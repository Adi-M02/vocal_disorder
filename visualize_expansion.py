import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import addcopyfighandler

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
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        mask = encoded["attention_mask"].unsqueeze(-1).expand_as(token_embeddings).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normed.cpu().numpy()

# Terms
symptom_terms = [
    'Vomit air', 'Acid reflux', 'Air sick', 'Air burp', 'Air vomiting',
    'Croaks', 'Chest pain', 'Gassy', 'Gurgling', 'Gurgles', 'Croaking',
    'Internal burps', 'Bloating', 'Retching', 'Reflux', 'Regurgitation',
    'Shortness of breath', 'Throwing up', 'Throat tightness', 'Throat gurgles',
    'Hiccups', 'Supragastric belching', 'Indigestion', 'Difficulty breathing',
    'Gastrointestinal distress'
]

other_terms = [
    'globus', 'gag_usually', 'hiccups maybe', 'burp fear vomiting',
    'air vomit makes', 'vomit sorry', 'extreme abdominal distension',
    'bloating main symptoms', 'gurgles eating', 'hear stomach gurgling',
    'stomach pain bloating', 'gas trapped', 'pressure neck',
    'body bloating', 'vomiting loud'
]

all_terms = symptom_terms + other_terms

# Compute embeddings
embeddings = embed_terms(all_terms, tokenizer, model, device=device)

# PCA reduction
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Assign colors: blue for symptom_terms, red for others
colors = ['blue' if t in symptom_terms else 'red' for t in all_terms]

# Plot PCA scatter
plt.figure(figsize=(10, 6))
for (x, y), term, c in zip(reduced, all_terms, colors):
    size = 100 if term in symptom_terms else 60
    plt.scatter(x, y, c=c, s=size, edgecolors='k', linewidths=0.5)
    plt.text(x + 0.01, y + 0.01, term, fontsize=9)

# Create legend handles
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Symptom terms',
           markerfacecolor='blue', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Other terms',
           markerfacecolor='red', markersize=8, markeredgecolor='k')
]

plt.legend(handles=legend_elements, loc='best')
plt.title("PCA of Symptom vs. Other Terms")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
