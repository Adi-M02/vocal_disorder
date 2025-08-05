"file with function to verify lemma coverage on tokenized documents and to combine lemmas into one json"
import sys
import json

sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from utils.load_lemmatizer import load_lookup
from spellchecker_folder.spellchecker import spellcheck_token_list

def combine_lemma_jsons(agree_json, flagged_json, output_json):
    with open(agree_json, 'r') as f:
        agree_data = json.load(f)
    with open(flagged_json, 'r') as f:
        flagged_data = json.load(f)

    combined_data = agree_data | flagged_data

    with open(output_json, 'w') as f:
        json.dump(combined_data, f, indent=2)

def verify_lemma_coverage():
    lookup_map = load_lookup('testing/combined_lemmas.json')
    missing_lemmas = set()
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    for text in docs:
        tokens = clean_and_tokenize(text)
        tokens = [lookup_map.get(tok, tok) for tok in tokens]
        tokens = spellcheck_token_list(tokens)
        for token in tokens:
            if token not in lookup_map:
                missing_lemmas.add(token)
    print(f"Missing lemmas: {len(missing_lemmas)}")
    with open('testing/missing_lemmas.json', 'w') as f:
        json.dump({token: token for token in missing_lemmas}, f, indent=2)

if __name__ == "__main__":
    verify_lemma_coverage()