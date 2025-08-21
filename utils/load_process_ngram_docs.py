import sys
sys.path.append("../vocal_disorder")
from utils.load_and_process_docs import process_all_noburp
from utils.text_pipeline import remove_unigram_stopwords
from testing.test_ngram_generation import load_phrasers_from_dir, apply_ngrams  

def process_ngram_docs(ngram_phraser_dir):
    docs = process_all_noburp(stoplist=False)
    bigram, trigram = load_phrasers_from_dir(ngram_phraser_dir)
    out = []
    for doc in docs:
        doc = apply_ngrams(doc, (bigram, trigram))
        doc = remove_unigram_stopwords(doc)
        out.append(doc)
    return out
