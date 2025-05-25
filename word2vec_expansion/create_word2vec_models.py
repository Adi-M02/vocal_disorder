import logging
from gensim.models import Word2Vec
import sys

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize


docs = return_documents("reddit", "noburp_all", ["noburp"])
print(f"Number of documents: {len(docs)}")
cleaned_docs = []
for doc in docs:
    # Tokenize and clean the text
    tokens = clean_and_tokenize(doc["noburp"])
    cleaned_docs.append(tokens)
print(f"Number of documents: {len(cleaned_docs)}")