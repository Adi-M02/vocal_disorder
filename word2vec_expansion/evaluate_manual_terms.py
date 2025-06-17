import sys
import json
from typing import List, Optional, Tuple
from collections import Counter

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list
from tqdm import tqdm

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

    term_counter = Counter()
    for text in tqdm(docs, desc="Processing docs"):
        toks = token_fn(text)
        # apply lemma lookup
        lemtoks = [lookup_map.get(tok, tok) for tok in toks]
        for term in lemtoks:
            term_counter[term] += 1
    print(f"Total unique terms in docs: {len(term_counter)}")
    
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
            if term_counter[term] < 5: 
                print(term, term_counter[term])
        else:
            print(term, 0)

if __name__ == "__main__":
    main()