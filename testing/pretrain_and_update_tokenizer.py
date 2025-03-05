import html
import re
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM

def clean_and_tokenize(text):
    """
    Cleans the text by unescaping HTML, removing URLs, special characters,
    and extra spaces, then tokenizes it.
    """
    # Unescape HTML characters
    text = html.unescape(text)
    # Remove URLs, special characters, and extra spaces; convert to lowercase
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    # Tokenize using nltk's word_tokenize
    tokens = word_tokenize(text)
    return tokens

def expand_vocabulary(corpus, tokenizer, model, min_freq=3):
    """
    Given a list of raw text posts (corpus), this function:
      1. Cleans and tokenizes each post.
      2. Counts token frequencies.
      3. Identifies tokens that are out-of-vocabulary (OOV) or tokenized as [UNK].
      4. Adds those tokens to the tokenizer.
      5. Resizes the model's token embeddings.
    
    Args:
      corpus (list): List of raw text posts.
      tokenizer (PreTrainedTokenizer): The model's tokenizer.
      model (PreTrainedModel): The model.
      min_freq (int): Minimum frequency for a token to be considered.
      
    Returns:
      Updated tokenizer and model.
    """
    # Count frequency of tokens across the corpus
    token_counter = Counter()
    for text in corpus:
        tokens = clean_and_tokenize(text)
        token_counter.update(tokens)
    
    # Only consider candidate tokens that appear at least min_freq times
    candidate_tokens = {token for token, count in token_counter.items() if count >= min_freq}
    
    # Get existing vocabulary from tokenizer
    existing_vocab = set(tokenizer.get_vocab().keys())
    
    oov_tokens = set()
    for token in candidate_tokens:
        # Tokenize the candidate word
        tokenized = tokenizer.tokenize(token)
        # If tokenization yields nothing or just the unknown token, consider it OOV
        if not tokenized or (len(tokenized) == 1 and tokenized[0] == tokenizer.unk_token):
            oov_tokens.add(token)
        # Also add if it's not directly in the vocabulary
        elif token not in existing_vocab:
            oov_tokens.add(token)
    
    print("Out-of-vocabulary tokens to add:", oov_tokens)
    
    # Add the new tokens to the tokenizer
    num_added = tokenizer.add_tokens(list(oov_tokens))
    print(f"Added {num_added} new tokens to the tokenizer.")
    
    # Resize the model's embeddings to account for the new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

# --------------------------
# Example Usage
# --------------------------

# Load the base BioClinicalBERT tokenizer and model
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Assume 'corpus' is your list of raw NoBurp posts
corpus = [
    "I have RCPD and I can't burp normally.",
    "The treatment involves botox injections and specialized exercises.",
    "Patients often experience gurgling and abnormal swallow patterns.",
    # ... add all your posts here
]

# Expand the vocabulary based on your corpus
tokenizer, model = expand_vocabulary(corpus, tokenizer, model, min_freq=2)

# # Optionally, save the updated tokenizer and model for future use
# output_dir = "bioclinicalbert_updated"
# tokenizer.save_pretrained(output_dir)
# model.save_pretrained(output_dir)
