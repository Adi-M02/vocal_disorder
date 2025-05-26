import logging
import sys
import os
import datetime
import time
import json
import re
import unicodedata

from gensim.models import Word2Vec

# allow you to import your Mongo helper and tokenizer
sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

def load_terms(path: str) -> dict[str, list[str]]:
    """
    Load category→list-of-terms from JSON, replacing underscores with spaces.
    """
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }

def main():
    # 1) load raw documents
    docs = return_documents("reddit", "noburp_all", ["noburp"])
    print(f"Number of documents: {len(docs)}")

    # 2) clean and tokenize your documents
    cleaned_docs: list[list[str]] = []
    for doc in docs:
        tokens = clean_and_tokenize(doc)
        cleaned_docs.append(tokens)
    print(f"Number of tokenized documents: {len(cleaned_docs)}")

    # 3) load & tokenize custom terms so they end up in vocab
    custom_terms_map = load_terms("rcpd_terms.json")
    custom_token_lists: list[list[str]] = []
    for terms in custom_terms_map.values():
        for term in terms:
            toks = clean_and_tokenize(term)
            if toks:
                custom_token_lists.append(toks)
    print(f"Loaded {sum(len(v) for v in custom_terms_map.values())} custom terms → {len(custom_token_lists)} token lists")

    # 4) build timestamped output directory
    now = datetime.datetime.now()
    out_dir = now.strftime("word2vec_expansion/word2vec_%m_%d_%H_%M")
    os.makedirs(out_dir, exist_ok=True)

    # 5) train CBOW (with custom-vocab update)
    start_cbow = time.time()
    cbow = Word2Vec(
        vector_size=300,
        window=7,
        min_count=5,
        sg=0,
        workers=os.cpu_count() - 4
    )
    # build on corpus vocabulary first
    cbow.build_vocab(cleaned_docs)
    # merge custom tokens x 5 to meet min_count threshold
    cbow.build_vocab(custom_token_lists*5, update=True)
    # now train
    cbow.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=5)
    cbow.save(os.path.join(out_dir, "word2vec_cbow.model"))
    print(f"CBOW training took {time.time() - start_cbow:.2f} seconds")

    # 6) train Skip-gram (with custom-vocab update)
    start_skip = time.time()
    skipgram = Word2Vec(
        vector_size=300,
        window=7,
        min_count=5,
        sg=1,
        workers=os.cpu_count() - 4
    )
    skipgram.build_vocab(cleaned_docs)
    skipgram.build_vocab(custom_token_lists*5, update=True)
    skipgram.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=5)
    skipgram.save(os.path.join(out_dir, "word2vec_skipgram.model"))
    print(f"Skip-gram training took {time.time() - start_skip:.2f} seconds")

    print(f"Models saved to {out_dir}")

if __name__ == "__main__":
    main()