from transformers import AutoTokenizer, AutoModel
import torch
from nltk.corpus import stopwords
from nltk import download
import string
import torch.nn.functional as F
import sys 
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize

# Load model
finetuned_dir = "vocab_expansion_with_other_models/finetuned_modernBERT-base_06_11_13_48"
tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
model = AutoModel.from_pretrained(finetuned_dir)

# List of terms
terms_list = [
        "Throat muscles",
        "GI",
        "Cricoid",
        "Swallow",
        "Peristalsis",
        "Retrograde Cricopharyngeal Dysfunction"
]

# Compute average embedding for the list
term_embeddings = []
for term in terms_list:
    inputs = tokenizer(term, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze(0))
    indices = [i for i, tok in enumerate(tokens) if tok not in tokenizer.all_special_tokens]
    term_emb = embeddings[indices].mean(dim=0)
    term_embeddings.append(term_emb)
average_embedding = torch.stack(term_embeddings).mean(dim=0)

# Target sentence
# Download stopwords if needed
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    download('stopwords')
    stop_words = set(stopwords.words('english'))

original_sentence = '''
Today was the day! 

Yay!

I'm just so happy! 

Super cool. Went in using a long needle straight through my throat - I was awake. She first numbed my skin, then injected some number under my skin, then used a big needle (did not feel) to go straight into my throat and numb. You instantly cough as a reflex of that muscle being frozen. Then she sticks the giant needle into your throat using an ultrasound. She needs to access the muscle behind the swallowing so she asks you to make an extra gentle sniff with the tip of your nose. 

Using ultrasound sound she can hear when she's in the right place. I had to take a big swallow with the needle in my throat. She wasn't impressed but it took me a second of breathing to restabilize and I was good to go! She had me sniff and put the needle deeper. Had me sniff again. Deeper. Again deeper. Until we got silence on the ultrasound. Then she asked me hold still and she injected the Botox over 15 seconds. Felt like a marble in my throat for a minute or two.

After I left I swallowed a bunch and an hour later felt totally ok. Can't sing but can talk normal. 

Takes 3-5 days to start working. 90% of patients respond to dose (50unit) but some patients require 100.

I follow up in a month and 3 months. 

I have issues with my LES too, she said let's see how this goes. This could cure my LES. This could also make me have heart burn for 2 months while the botox wears off because of my LES - in which case she said I probably need LES treatment.
'''

# Remove stopwords and punctuation
words = original_sentence.lower().split()
filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]
sentence = " ".join(filtered_words)
sentence = clean_and_tokenize(sentence)  # Use the custom tokenizer
sentence = " ".join(sentence)  # Join tokens back into a sentence
# print(f"Original: {original_sentence}")
# print(f"Filtered: {sentence}")
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state.squeeze(0)
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze(0))
# Remove special tokens
tokens = tokens[1:-1]
embeddings = embeddings[1:-1]

# Generate 1,2,3-grams
ngram_embeddings = {}
for n in [1, 2, 3]:
    for i in range(len(tokens) - n + 1):
        ngram_tokens = tokens[i:i+n]
        ngram_text = tokenizer.convert_tokens_to_string(ngram_tokens)
        ngram_emb = embeddings[i:i+n].mean(dim=0)
        ngram_embeddings[ngram_text] = ngram_emb

# Compute cosine similarity to average embedding
results = []
for ngram_text, ngram_emb in ngram_embeddings.items():
    sim = F.cosine_similarity(ngram_emb.unsqueeze(0), average_embedding.unsqueeze(0)).item()
    results.append((ngram_text, sim))

# Sort and print
results.sort(key=lambda x: x[1], reverse=True)
for ngram_text, sim in results:
    print(f"{ngram_text}: {sim:.4f}")
