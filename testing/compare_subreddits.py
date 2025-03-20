import nltk
import numpy as np
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
import sys
import html
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("/local/disk2/not_backed_up/amukundan/research/vocal_disorder")
import query_mongo as query
import initial_analyses as ia

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

MAIN_RCPD_SUBREDDITS = ["noburp"]
SYMPTOM_DISEASE_SUBREDDITS = ["emetophobia", "gerd", "ibs", "sibo", "emetophobiarecovery", "pots", "gastritis"]
SELF_HELP_MEDICATION_SUBREDDITS = ["trees", "advice", "drugs", "supplements", "shrooms"]
OTHER_SUBREDDITS = ["anxiety", "healthanxiety"]

def analyze_subreddits(subreddit_group, label):
    print(f"\n{'='*40}\nAnalyzing: {label}\n{'='*40}")

    # Query posts from MongoDB
    raw_posts = query.get_posts_by_subreddits(subreddit_group)

    if not raw_posts:
        print("No posts found.")
        return

    # Preprocess text
    processed_posts = [ia.preprocess_text(post['selftext']) for post in raw_posts]

    # 1️⃣ **LSA Analysis**
    topics = ia.apply_lsa(processed_posts, 5)
    print("\nLSA Topics Found:")
    for topic in topics:
        print(topic)

    # 2️⃣ **TF-IDF Analysis**
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(processed_posts)
    feature_names = vectorizer.get_feature_names_out()

    top_n = 20
    sorted_tfidf = sorted(zip(vectorizer.idf_, feature_names))[:top_n]
    print("\nTop TF-IDF Terms:", [word for _, word in sorted_tfidf])

    # 3️⃣ **N-Gram Analysis (Bigrams & Trigrams)**
    def get_top_ngrams(corpus, n=2, top_n=10):
        all_ngrams = [ngram for text in corpus for ngram in ngrams(text.split(), n)]
        ngram_counts = Counter(all_ngrams)
        return ngram_counts.most_common(top_n)

    print("\nTop Bigrams:", get_top_ngrams(processed_posts, n=2))
    print("Top Trigrams:", get_top_ngrams(processed_posts, n=3))

    # 4️⃣ **LDA Topic Modeling**
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_matrix = lda.fit_transform(tfidf_matrix)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_words = [terms[j] for j in comp.argsort()[-10:]]
        topics.append(f"Topic {i+1}: {', '.join(top_words)}")

    print("\nLDA Topics Found:")
    for topic in topics:
        print(topic)

    # 5️⃣ **Sentiment Analysis**
    sentiments = [sia.polarity_scores(post)["compound"] for post in processed_posts]
    avg_sentiment = np.mean(sentiments)
    print(f"\nAverage Sentiment Score: {avg_sentiment:.3f}")

    # 6️⃣ **Engagement Analysis**
    avg_score = np.mean([post.get("score", 0) for post in raw_posts])
    avg_comments = np.mean([post.get("num_comments", 0) for post in raw_posts])
    avg_post_length = np.mean([len(html.unescape(post["selftext"]).split()) for post in raw_posts])

    print(f"\nEngagement Metrics:\nAvg Upvotes: {avg_score:.2f}\nAvg Comments: {avg_comments:.2f}\nAvg Post Length: {avg_post_length:.2f} words")

# analyze_subreddits(MAIN_RCPD_SUBREDDITS, "Main RCPD Subreddits")
# analyze_subreddits(SYMPTOM_DISEASE_SUBREDDITS, "Symptom/Related Disease Subreddits")
# analyze_subreddits(SELF_HELP_MEDICATION_SUBREDDITS, "Self-Help/Medication Subreddits")

# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Convert posts into embeddings
# def get_embeddings(posts):
#     return model.encode(posts, convert_to_tensor=True)

# processed_main_posts = [ia.preprocess_text(post['selftext']) for post in query.get_posts_by_subreddits(MAIN_RCPD_SUBREDDITS)]
# processed_symptom_posts = [ia.preprocess_text(post['selftext']) for post in query.get_posts_by_subreddits(SYMPTOM_DISEASE_SUBREDDITS)]
# processed_selfhelp_posts = [ia.preprocess_text(post['selftext']) for post in query.get_posts_by_subreddits(SELF_HELP_MEDICATION_SUBREDDITS)]

# # Get embeddings for each subreddit category (ensuring tensor format)
# main_embeddings = get_embeddings(processed_main_posts).cpu().numpy()
# symptom_embeddings = get_embeddings(processed_symptom_posts).cpu().numpy()
# selfhelp_embeddings = get_embeddings(processed_selfhelp_posts).cpu().numpy()

# # Compute mean cosine similarity between subreddit categories
# main_vs_symptom = np.mean(cosine_similarity(main_embeddings, symptom_embeddings))
# main_vs_selfhelp = np.mean(cosine_similarity(main_embeddings, selfhelp_embeddings))
# symptom_vs_selfhelp = np.mean(cosine_similarity(symptom_embeddings, selfhelp_embeddings))

# # Print semantic similarity scores
# print(f"Semantic Similarity (Main vs Symptom): {main_vs_symptom:.3f}")
# print(f"Semantic Similarity (Main vs Self-Help): {main_vs_selfhelp:.3f}")
# print(f"Semantic Similarity (Symptom vs Self-Help): {symptom_vs_selfhelp:.3f}")

fasttext.util.download_model('en', if_exists='ignore')  # English embeddings

# Load FastText model
ft_model = fasttext.load_model('cc.en.300.bin')  # 300-dimensional English embeddings

# Get vector for a word
def get_word_vector(word):
    return ft_model.get_word_vector(word)

# Compute cosine similarity between words
def compute_word_similarity(word1, word2):
    vec1, vec2 = get_word_vector(word1), get_word_vector(word2)
    return cosine_similarity([vec1], [vec2])[0][0]

# Example comparisons
print(f"Similarity between 'burping' and 'throat': {compute_word_similarity('burping', 'throat'):.3f}")
print(f"Similarity between 'botox' and 'treatment': {compute_word_similarity('botox', 'treatment'):.3f}")
print(f"Similarity between 'pain' and 'recovery': {compute_word_similarity('pain', 'recovery'):.3f}")