import torch
import numpy as np
from sentence_transformers import util
import re
from collections import Counter
import query_mongo as query
import html
import os
import nltk
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)
nltk.data.path.append(NLTK_DATA_DIR)
nltk_resources = ["stopwords", "punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger_eng"]
for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel

TERM_CATEGORY_DICT = {
    "Experience with RCPD": ['Faux pas', 'Anxiety', 'Social-anxiety', 'Flare-ups', 'Misdiagnosis', 'R-CPD', 'Isolation'], 
    "Symptoms": ['Vomit', 'Vomit air', 'Acid reflux', 'Air sick', 'Air burp', 'Air vomiting', 'Cough', 'Croaks', 'Chest pain', 'Gag', 'Gassy', 'Gurgling', 'Gurgles', 'Gas', 'Croaking', 'Internal burps', 'Pressure', 'Bloating', 'Retching', 'Reflux', 'Regurgitation', 'Symptoms', 'Shortness of breath', 'Throwing up', 'Throat tightness', 'Throat gurgles', 'Hiccups', 'Supragastric belching', 'Indigestion', 'Difficulty breathing', 'Gastrointestinal distress'],
    "Self-treatment methods": ['Chamomile', 'Tea', 'Exercises', 'Gas-X', 'Famotidine', 'Fizzy drinks', 'Omeprazole', 'Neck turning', 'Self-taught', 'Self-curing', 'Shaker', 'Pelvic tilt', 'Mounjaro', 'antacids', 'kiss the ceiling', 'Self-cured', 'Rapid Drink Challenge'],
    "Doctor based interventions": ['(Pre,post) -Botox', 'In-office procedure', 'surgery', 'Anesthesia', 'procedure', 'Units', 'Xeomin', 'Esophageal dilation', 'Injections', 'Saline'],
    "Associated possible conditions": ['hiatal hernia', 'Dyspepsia', 'GERD', 'Emetophobia', 'Abdomino phrenic dyssynergia (APD)', 'GI disorder', 'irritable bowel syndrome'], 
    "Who to seek for diagnosis/treatment": ['ENT', 'Gastroenterologist', 'Laryngologist', 'Otolaryngologist', 'PCP', 'Specialist', 'Insurance'], 
    "Diagnostic Steps": ['Endoscopy', 'Nasoendoscopy', 'Swallow tests', 'esophageal examination', 'HRM', 'Manometry', 'Fluoroscopy', 'Imaging', 'Barium swallow', 'FEES test'],
    "General Anatomy and Physiology Involved": ['Throat muscles', 'GI', 'Cricoid', 'Swallow', 'Peristalsis', 'Retrograde Cricopharyngeal Dysfunction']
}
MAIN_RCPD_SUBREDDITS = ["noburp"]

# Load English stopwords
STOPWORDS = set(stopwords.words("english"))

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character for WordNet lemmatization"""
    from nltk import pos_tag
    tag = pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun

def preprocess_text(text):
    """
    Cleans and preprocesses text by removing stopwords and tokenizing (without lemmatization).

    :param text: Raw text to preprocess.
    :return: Preprocessed text as a single string.
    """
    # Unescape HTML characters
    text = html.unescape(text)

    # Remove special characters, URLs, and extra spaces; convert to lowercase
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())

    # Tokenization
    words = word_tokenize(text)

    # Stopword removal (without lemmatization)
    processed_tokens = [word for word in words if word not in STOPWORDS]

    return " ".join(processed_tokens)

def preprocess_text_lemmatize(text):
    """
    Cleans and preprocesses text by removing stopwords and tokenizing (with lemmatization).

    :param text: Raw text to preprocess.
    :return: Preprocessed text as a single string.
    """
    # Unescape HTML characters
    text = html.unescape(text)

    # Remove special characters, URLs, and extra spaces; convert to lowercase
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())

    # Tokenization
    words = word_tokenize(text)

    # Stopword removal (with lemmatization)
    processed_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in STOPWORDS]
    
    return " ".join(processed_tokens)

def load_and_preprocess_posts(subreddits, query_module, preprocess_function):
    """
    Loads posts from the specified subreddits using the provided query module
    
    Parameters:
        subreddits (list): List of subreddit names to query.
        query_module: Module (or object) that contains the get_posts_by_subreddits function.
        
    Returns:
        list: A list of processed post texts.
    """
    raw_posts = query_module.get_posts_by_subreddits(subreddits, collection_name="noburp_posts")
    processed_posts = [preprocess_function(post['selftext']) for post in raw_posts]
    return processed_posts

def tokenize(text):
    """
    Lowercases the text and extracts alphanumeric tokens (including apostrophes).
    
    Parameters:
        text (str): Input text.
        
    Returns:
        list: List of tokens.
    """
    return re.findall(r"\b[\w']+\b", text.lower())

def generate_candidate_terms(processed_posts, min_freq_unigram=5, min_freq_bigram=3, min_freq_trigram=2):
    """
    Generates candidate terms from processed posts by computing unigrams, bigrams, and trigrams
    and applying frequency thresholds.
    
    Parameters:
        processed_posts (list): A list of preprocessed post texts.
        min_freq_unigram (int): Minimum frequency threshold for unigrams.
        min_freq_bigram (int): Minimum frequency threshold for bigrams.
        min_freq_trigram (int): Minimum frequency threshold for trigrams.
        
    Returns:
        list: A list of unique candidate terms.
    """
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    for doc in processed_posts:
        tokens = tokenize(doc)
        unigram_counter.update(tokens)
        
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i+1]
            bigram_counter.update([bigram])
        
        for i in range(len(tokens) - 2):
            trigram = tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2]
            trigram_counter.update([trigram])
    
    candidate_unigrams = [w for w, f in unigram_counter.items() if f >= min_freq_unigram]
    candidate_bigrams  = [w for w, f in bigram_counter.items()  if f >= min_freq_bigram]
    candidate_trigrams = [w for w, f in trigram_counter.items() if f >= min_freq_trigram]
    
    candidate_terms = list(set(candidate_unigrams + candidate_bigrams + candidate_trigrams))
    
    print(f"Unigrams (freq >= {min_freq_unigram}): {len(candidate_unigrams)}")
    print(f"Bigrams (freq >= {min_freq_bigram}):  {len(candidate_bigrams)}")
    print(f"Trigrams (freq >= {min_freq_trigram}): {len(candidate_trigrams)}")
    print(f"Total candidate terms: {len(candidate_terms)}")
    
    return candidate_terms
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

def expand_terms_category_dict(term_category_dict, tokenizer, model, candidate_embeddings, top_n=10, device="cpu"):
    """
    Expands each category in term_category_dict by finding the top_n most similar candidate terms
    (based on cosine similarity between embeddings) to the average embedding of the seed terms in that category.
    
    Parameters:
        term_category_dict (dict): Dictionary mapping category names to lists of seed terms.
        tokenizer: Transformers tokenizer (e.g., AutoTokenizer.from_pretrained(...)).
        model: Transformers model (e.g., AutoModel.from_pretrained(...)).
        candidate_embeddings (dict): Dictionary mapping each candidate term (string) to its embedding (numpy array).
        top_n (int): Number of candidate terms to select per category.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        
    Returns:
        dict: A dictionary mapping each category to a list of terms comprising the union of the original seed terms 
              and the top_n expanded candidate terms for that category.
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
            score = util.pytorch_cos_sim(cat_emb_tensor, cand_emb_tensor).item()  # cosine similarity
            sim_list.append((cand_term, score))
        # Sort candidate terms by descending similarity and select top_n
        sim_list.sort(key=lambda x: x[1], reverse=True)
        expanded_terms[cat] = [term for term, score in sim_list[:top_n]]
    
    # 5. For each category, merge the original seed terms and the expanded terms.
    term_category_dict_expanded = {}
    for cat in term_category_dict:
        original = set(term_category_dict[cat])
        new_terms = set(expanded_terms.get(cat, []))
        term_category_dict_expanded[cat] = list(original.union(new_terms))
    
    return term_category_dict_expanded

def update_term_category_dict(original_dict, expanded_dict):
    """
    Merges the original term category dictionary with an expanded dictionary.
    For each category, the returned dictionary contains the union of the original seed terms
    and any new (expanded) terms. If the expanded dictionary contains a category not present
    in the original, that category is added.
    
    Parameters:
        original_dict (dict): The original dictionary mapping category names to lists of seed terms.
        expanded_dict (dict): A dictionary mapping category names to lists of expanded terms.
    
    Returns:
        dict: A new dictionary with updated categories.
    """
    updated_dict = original_dict.copy()
    for cat, new_terms in expanded_dict.items():
        if cat in updated_dict:
            # Merge existing terms with the new terms
            updated_dict[cat] = list(set(updated_dict[cat]).union(new_terms))
        else:
            # If this category wasn't in the original, add it
            updated_dict[cat] = new_terms
    return updated_dict

def expand_category_terms_pipeline(subreddits,
                                   query_module,
                                   preprocess_function,
                                   min_freq_unigram=5,
                                   min_freq_bigram=3,
                                   min_freq_trigram=2,
                                   model_name="emilyalsentzer/Bio_ClinicalBERT",
                                   convert_underscores=False,
                                   batch_size=64,
                                   term_category_dict=None,
                                   top_n=10,
                                   device="cuda"):
    """
    Full pipeline to expand a term category dictionary using candidate terms from posts.
    
    Parameters:
        subreddits (list): List of subreddit names to query.
        query_module: Module (or object) that provides get_posts_by_subreddits().
        preprocess_function: Function to preprocess post text (e.g., preprocess_text).
        min_freq_unigram (int): Frequency threshold for unigrams.
        min_freq_bigram (int): Frequency threshold for bigrams.
        min_freq_trigram (int): Frequency threshold for trigrams.
        model_name (str): Name/path of the transformer model to use.
        convert_underscores (bool): If True, converts underscores in candidate terms to spaces before embedding.
        batch_size (int): Batch size for embedding candidate terms.
        term_category_dict (dict): Original term category dictionary mapping category names to lists of seed terms.
        top_n (int): Number of candidate terms to select per category for expansion.
        device (str): Device to run model on (e.g., "cpu" or "cuda").
        
    Returns:
        dict: An updated term category dictionary with expanded terms added to each category.
    """
    # 1. Load and preprocess posts.
    posts = load_and_preprocess_posts(subreddits, query_module, preprocess_function)
    
    # 2. Generate candidate terms.
    candidate_terms = generate_candidate_terms(posts,
                                                 min_freq_unigram=min_freq_unigram,
                                                 min_freq_bigram=min_freq_bigram,
                                                 min_freq_trigram=min_freq_trigram)
    
    # 3. Load the pretrained model and tokenizer.
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # inference mode
    
    # 4. Prepare candidate texts for embedding.
    if convert_underscores:
        candidate_texts_for_embedding = [term.replace("_", " ") for term in candidate_terms]
    else:
        candidate_texts_for_embedding = candidate_terms[:]
    
    # 5. Embed candidate terms in batches.
    candidate_embeddings = {}
    for i in range(0, len(candidate_texts_for_embedding), batch_size):
        batch = candidate_texts_for_embedding[i:i+batch_size]
        embs = embed_texts(batch, tokenizer, model, device=device)
        # Use the original candidate terms (with underscores preserved) as keys.
        for original_term, embedded_term in zip(candidate_terms[i:i+batch_size], embs):
            candidate_embeddings[original_term] = embedded_term
    
    # 6. Expand term categories by category.
    expanded_terms_by_category = expand_terms_category_dict(term_category_dict,
                                                                    tokenizer,
                                                                    model,
                                                                    candidate_embeddings,
                                                                    top_n=top_n,
                                                                    device=device)
    
    # 7. Update the original term category dict with the expanded terms.
    updated_term_category_dict = update_term_category_dict(term_category_dict,
                                                            expanded_terms_by_category)
    
    return updated_term_category_dict

if __name__ == "__main__":
    updated_dict = expand_category_terms_pipeline(subreddits=MAIN_RCPD_SUBREDDITS,
                                                query_module=query,
                                                preprocess_function=preprocess_text,
                                                min_freq_unigram=5,
                                                min_freq_bigram=3,
                                                min_freq_trigram=2,
                                                model_name="emilyalsentzer/Bio_ClinicalBERT",
                                                convert_underscores=True,
                                                batch_size=64,
                                                term_category_dict=TERM_CATEGORY_DICT,
                                                top_n=20,
                                                device="cuda")
    for cat, terms in updated_dict.items():
        print(f"{cat}: {terms}")
