import sys
import json
from typing import List, Optional, Tuple
from collections import Counter

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list
from tqdm import tqdm
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def extract_frequent_ngrams(
    max_ngram: int,
    tok_fn,
    lookup_map: dict
) -> list[str]:
    # fetch all documents
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        mongo_uri="mongodb://localhost:27017/"
    )
    
    counts = Counter()
    
    # slide an n-length window over each doc’s token list
    for doc in tqdm(docs, desc=f"Loading docs for ngrams"):
        tokens = [lookup_map.get(t, t) for t in tok_fn(doc)]
        L = len(tokens)
        for n in range(2, max_ngram + 1):
            if L < n:
                break
            for i in range(L - n + 1):
                gram = tuple(tokens[i:i + n])
                # # skip if first or last token is a stopword
                # if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                #     continue
                counts[gram] += 1

    # hardcoded min_count for 2-gram and 3-gram
    result = []
    for gram, cnt in counts.items():
        n = len(gram)
        if n == 2 and cnt >= 0:
            result.append(" ".join(gram))
        elif n == 3 and cnt >= 0:
            result.append(" ".join(gram))
    return result

def load_manual_terms(
    path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn
) -> List[str]:
    """
    Load comma-separated manual terms, tokenize and filter by ngram_filter.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().lower()
    candidates = [t.strip() for t in raw.split(",") if t.strip()]

    cleaned: List[str] = []
    for term in candidates:
        tokens = tok_fn(term)
        if not tokens:
            continue
        L = len(tokens)
        if ngram_filter is None or (ngram_filter[0] <= L <= ngram_filter[1]):
            cleaned.append(" ".join(tokens))
    return cleaned

def token_fn(text):
    return spellcheck_token_list(clean_and_tokenize(text))

def load_lookup(path: str) -> dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def main():
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    print(f"Number of documents fetched: {len(docs)}")

    lookup_map = load_lookup("testing/lemma_lookup.json")

    # 1) Count all unigrams as before
    term_counter = Counter()

    # 2) Get your frequent n-grams (strings) and turn into tuple form
    ngram_terms = extract_frequent_ngrams(
        max_ngram=3,
        tok_fn=token_fn,
        lookup_map=lookup_map
    )
    print(f"Extracted {len(ngram_terms)} frequent n-grams")
    ngram_set = {tuple(term.split()) for term in ngram_terms}

    # 3) Compute the maximum n ( here up to 3 )
    max_n = max(len(ngram) for ngram in ngram_set)

    # 4) Count only those n-grams via a sliding window
    for text in tqdm(docs, desc="Counting ngram terms"):
        toks    = token_fn(text)
        lemtoks = [lookup_map.get(tok, tok) for tok in toks]
        L       = len(lemtoks)

        for i in range(L):
            # try n = 2..max_n (skip 1 since unigrams are already counted)
            for n in range(2, max_n + 1):
                if i + n > L:
                    break
                gram = tuple(lemtoks[i : i + n])
                if gram in ngram_set:
                    ngram_key = " ".join(gram)
                    term_counter[ngram_key] += 1
                    for token in gram:
                        term_counter[token] += 1
    manual_terms = load_manual_terms(
        path="vocabulary_evaluation/updated_manual_terms_6_12/manual_terms.txt",
        ngram_filter=None,
        tok_fn=token_fn
    )
    print(f"Loaded {len(manual_terms)} manual terms")
    # Apply lemma lookup to all manual terms
    manual_terms = [ " ".join([lookup_map.get(tok, tok) for tok in term.split()]) for term in manual_terms ]

    for term in manual_terms:
        if term in term_counter:
            if term_counter[term] < 3: 
                print(term, term_counter[term])
        else:
            print(term, 0)

if __name__ == "__main__":
    main()