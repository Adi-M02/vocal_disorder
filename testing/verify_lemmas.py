"file with function to verify lemma coverage on tokenized documents and to combine lemmas into one json"
import sys
import json

sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from utils.load_lemmatizer import load_lookup
from spellchecker_folder.spellchecker import spellcheck_token_list

def combine_lemma_jsons(agree_json_path, flagged_json_path, output_json_path):
    """
    Combine two JSON files into one, and print any keys that appear in both inputs.
    agree_json_path:    path to the "agree" JSON file
    flagged_json_path:  path to the "flagged" JSON file
    output_json_path:   path where the combined JSON will be written
    """
    # Load input JSONs
    with open(agree_json_path, 'r', encoding='utf-8') as f:
        agree_data = json.load(f)
    with open(flagged_json_path, 'r', encoding='utf-8') as f:
        flagged_data = json.load(f)

    # Identify repeated keys
    repeated_keys = set(agree_data.keys()) & set(flagged_data.keys())
    if repeated_keys:
        print("Repeated keys found in both JSONs:")
        for key in sorted(repeated_keys):
            print(f" - {key}")
    else:
        print("No repeated keys found.")

    # Combine data (flagged_data will override agree_data on key collisions)
    combined_data = {**agree_data, **flagged_data}

    # Write out the combined JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"Combined JSON written to: {output_json_path}")

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
        text = ' '.join(tokens)
        tokens = clean_and_tokenize(text)
        tokens = spellcheck_token_list(tokens)
        for token in tokens:
            if token not in lookup_map:
                missing_lemmas.add(token)
    print(f"Missing lemmas: {len(missing_lemmas)}")
    # with open('testing/missing_lemmas.json', 'w') as f:
    #     json.dump({token: token for token in missing_lemmas}, f, indent=2)

if __name__ == "__main__":
    # combine_lemma_jsons('testing/combined_lemmas_new.json', 'testing/combined_lemmas.json', 'testing/combined_lemmas_new_new.json')
    verify_lemma_coverage()