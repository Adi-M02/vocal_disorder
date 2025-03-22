import torch
import numpy as np
from sentence_transformers import util

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

def expand_term_categories(term_category_dict, tokenizer, model, candidate_terms, candidate_embeddings, top_n=10, device="cpu"):
    """
    Expands each category in term_category_dict by finding the top_n most similar candidate terms
    (based on cosine similarity between embeddings) to the average embedding of the seed terms in that category.
    
    Parameters:
        term_category_dict (dict): Dictionary mapping category names to lists of seed terms.
        tokenizer: Transformers tokenizer (e.g., AutoTokenizer.from_pretrained(...)).
        model: Transformers model (e.g., AutoModel.from_pretrained(...)).
        candidate_terms (list): List of candidate terms (strings) considered for expansion.
        candidate_embeddings (dict): Dictionary mapping each candidate term (string) to its embedding (numpy array).
        top_n (int): Number of candidate terms to select per category.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        
    Returns:
        list: A list of unique terms comprising the union of the original seed terms and 
              the expanded candidate terms.
    """
    # 1. Get all unique seed terms from the term_category_dict.
    all_seed_terms = []
    for cat, terms in term_category_dict.items():
        all_seed_terms.extend(terms)
    unique_seed_terms = list(set(all_seed_terms))
    
    # 2. Embed the unique seed terms using your embed_texts function.
    seed_embeddings = embed_texts(unique_seed_terms, tokenizer, model, device=device)
    term2emb = {term: emb for term, emb in zip(unique_seed_terms, seed_embeddings)}
    
    # 3. Compute an average embedding for each category.
    category_embeddings = {}
    for cat, terms in term_category_dict.items():
        cat_vecs = [term2emb[t] for t in terms if t in term2emb]
        if cat_vecs:
            category_embeddings[cat] = np.mean(cat_vecs, axis=0)
        else:
            # Fallback: use a zero vector (assuming candidate_embeddings is nonempty)
            category_embeddings[cat] = np.zeros_like(next(iter(candidate_embeddings.values())))
    
    # 4. For each category, compute cosine similarity between its embedding and each candidate's embedding.
    expanded_terms = {}  # category -> list of candidate terms (expanded)
    for cat, cat_emb in category_embeddings.items():
        cat_emb_tensor = torch.tensor(cat_emb).unsqueeze(0)  # shape (1, hidden_dim)
        sim_list = []
        for cand_term, cand_emb in candidate_embeddings.items():
            cand_emb_tensor = torch.tensor(cand_emb).unsqueeze(0)
            score = util.pytorch_cos_sim(cat_emb_tensor, cand_emb_tensor).item()
            sim_list.append((cand_term, score))
        # Sort candidate terms by descending similarity and select top_n
        sim_list.sort(key=lambda x: x[1], reverse=True)
        expanded_terms[cat] = [term for term, score in sim_list[:top_n]]
    
    # 5. Merge the original seed terms and the expanded terms from all categories.
    expanded_union = set()
    for cat in term_category_dict:
        original = set(term_category_dict[cat])
        new_terms = set(expanded_terms.get(cat, []))
        expanded_union.update(original.union(new_terms))
    
    return list(expanded_union)