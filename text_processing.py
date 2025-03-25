import re
import html
import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# NLTK setup
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk_resources = ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]
for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS."""
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

def preprocess_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    words = word_tokenize(text)
    processed_tokens = [word for word in words if word not in STOPWORDS]
    return " ".join(processed_tokens)

def preprocess_text_lemmatize(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    processed_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
                        for word, tag in pos_tags if word not in STOPWORDS]
    return " ".join(processed_tokens)
