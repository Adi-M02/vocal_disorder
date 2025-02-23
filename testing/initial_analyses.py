import re
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import html
import sys
import os
from collections import Counter
from nltk.util import ngrams

sys.path.append("/local/disk2/not_backed_up/amukundan/research/vocal_disorder")
import query_mongo as query

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
    Cleans and preprocesses text by removing stopwords, tokenizing, and lemmatizing with NLTK.

    :param text: Raw text to preprocess.
    :return: Preprocessed text as a single string.
    """
    # Unescape HTML characters
    text = html.unescape(text)

    # Remove special characters, URLs, and extra spaces
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text.lower())

    # Tokenization
    words = word_tokenize(text)

    # Lemmatization & Stopword Removal
    processed_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in STOPWORDS]

    return " ".join(processed_tokens)


def apply_lsa(corpus, num_topics):
    """
    Applies Latent Semantic Analysis (LSA) to find related concepts.

    :param corpus: List of preprocessed text documents.
    :param num_topics: Number of topics to extract.
    :return: Extracted topics with top words.
    """
    # Convert text corpus into TF-IDF matrix
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Apply LSA using Singular Value Decomposition (SVD)
    lsa = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # Extract top words for each topic
    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lsa.components_):
        top_words = [terms[j] for j in np.argsort(comp)[-10:]]  # Top 10 words
        topics.append(f"Topic {i+1}: " + ", ".join(top_words))

    return topics


if __name__ == "__main__":
    subreddit = "noburp"
    raw_posts = query.get_posts_by_subreddit(subreddit)

    if not raw_posts:
        print("No posts found for the given subreddit.")
    else:
        # Preprocess all posts
        processed_posts = [preprocess_text(post['selftext']) for post in raw_posts]

        # Apply LSA to find related concepts
        topics = apply_lsa(processed_posts, 5)

        # Print discovered topics
        print("\nLSA Topics Found:")
        for topic in topics:
            print(topic)

        vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(processed_posts)
        feature_names = vectorizer.get_feature_names_out()

        # Get the top TF-IDF terms
        top_n = 20
        sorted_tfidf = sorted(zip(vectorizer.idf_, feature_names))[:top_n]
        print("Top TF-IDF Terms:", [word for _, word in sorted_tfidf])

        def get_top_ngrams(corpus, n=2, top_n=10):
            all_ngrams = [ngram for text in corpus for ngram in ngrams(text.split(), n)]
            ngram_counts = Counter(all_ngrams)
            return ngram_counts.most_common(top_n)

        # Find common bigrams (2-word phrases)
        print("Top Bigrams:", get_top_ngrams(processed_posts, n=2))

        # Find common trigrams (3-word phrases)
        print("Top Trigrams:", get_top_ngrams(processed_posts, n=3))

        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_matrix = lda.fit_transform(tfidf_matrix)

        # Get the top words for each topic
        terms = vectorizer.get_feature_names_out()
        topics = []
        for i, comp in enumerate(lda.components_):
            top_words = [terms[j] for j in comp.argsort()[-10:]]
            topics.append(f"Topic {i+1}: {', '.join(top_words)}")

        print("\nLDA Topics Found:")
        for topic in topics:
            print(topic)
