import sys
import json

sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from utils.text_pipeline import process_text


users = "vocabulary_evaluation/manual_terms_7_12/users.txt"
annotated_file = "vocabulary_evaluation/manual_terms_7_12/5_min_count_terms/1_gram_seed_terms.json"
with open(users, "r") as f:
    user_list = [user.strip() for user in f.read().split(",") if user.strip()]

def test_return_annotated_users():
    for i, user in enumerate(user_list):
        print(f"Testing user {i}: {user}")
        docs = return_documents("reddit", "noburp_all", filter_users=[user])
        print(len(docs))

def test_annotated_set():
    docs = return_documents("reddit", "noburp_all", filter_users=user_list)
    with open(annotated_file, "r") as f:
        annotated_set = json.load(f)
    terms = set()
    for text in docs:
        toks = process_text(text)
        terms.update(toks)
    for annotated_term in annotated_set:
        processed_terms = process_text(annotated_term)
        for term in processed_terms:
            if term not in terms:
                print(term)

def check_for_empty_docs():
    docs = return_documents("reddit", "noburp_all", filter_users=user_list)
    for text in docs:
        text = process_text(text)
        if len(text) == 0:
            print("Empty document found:", text)
    print("No empty documents found.")
    
if __name__ == "__main__":
    check_for_empty_docs()