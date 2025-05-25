import logging
import sys
import os
import datetime
import time
from gensim.models import Word2Vec

sys.path.append('../vocal_disorder')

from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# 1) load raw documents
docs = return_documents("reddit", "noburp_all", ["noburp"])
print(f"Number of documents: {len(docs)}")

# 2) clean and tokenize
cleaned_docs = []
for doc in docs:
    tokens = clean_and_tokenize(doc)
    cleaned_docs.append(tokens)
print(f"Number of tokenized documents: {len(cleaned_docs)}")

# 3) build timestamped output directory
now = datetime.datetime.now()
out_dir = now.strftime("word2vec_expansion/word2vec_%m_%d_%H_%M")
os.makedirs(out_dir, exist_ok=True)

# 4) train CBOW
start_cbow = time.time()
cbow = Word2Vec(
    sentences=cleaned_docs,
    vector_size=300,
    window=7,
    min_count=5,
    sg=0,
    workers=os.cpu_count()-4,
    epochs=5
)
cbow.save(os.path.join(out_dir, "word2vec_cbow.model"))
end_cbow = time.time()
print(f"CBOW training took {end_cbow - start_cbow:.2f} seconds")

# 5) train Skip-gram
start_skip = time.time()
skipgram = Word2Vec(
    sentences=cleaned_docs,
    vector_size=300,
    window=7,
    min_count=5,
    sg=1,
    workers=os.cpu_count()-4,
    epochs=5
)
skipgram.save(os.path.join(out_dir, "word2vec_skipgram.model"))
end_skip = time.time()
print(f"Skip-gram training took {end_skip - start_skip:.2f} seconds")

print(f"Models saved to {out_dir}")
