import sys
import pytest
import json
import mongomock

# allow importing your modules
sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from utils.load_lemmatizer import load_lookup
import spellchecker_folder.spellchecker as sc
from spellchecker_folder.spellchecker import spellcheck_token_list

# Dummy SPELL checker for controlled suggestions
token_corrections = {'coughng': 'coughing'}
class DummySpell:
    def __init__(self, mapping):
        self.mapping = mapping
        self.called = []
    def correction(self, token):
        self.called.append(token)
        return self.mapping.get(token)

@pytest.fixture(autouse=True)
def patch_mongo(monkeypatch):
    import query_mongo
    client = mongomock.MongoClient()
    monkeypatch.setattr(query_mongo.pymongo, 'MongoClient', lambda uri: client)
    return client

def test_full_pipeline(tmp_path, patch_mongo):
    client = patch_mongo
    # -- Step A: populate fake DB --
    db = client['reddit']
    coll = db['noburp_all']
    coll.insert_many([
        {"body": "I was bloated and coughng", "subreddit": "noburp", "author": "u1"},
        {"title": "Gas and burping", "selftext": "Acid reflux and nausea", "subreddit": "noburp", "author": "u2"},
        {"body": "Ignore me if wrong sub", "subreddit": "other", "author": "u3"},
    ])

    # -- Step 1: fetch documents --
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    assert docs == [
        "I was bloated and coughng",
        "Gas and burping",
        "Acid reflux and nausea",
    ]

    # -- Step 2: clean & tokenize --
    token_lists = [clean_and_tokenize(text) for text in docs]
    assert token_lists == [
        ["i","was","bloated","and","coughng"],
        ["gas","and","burping"],
        ["acid","reflux","and","nausea"],
    ]

    # -- Step 3: lemmatize via lookup using real lemma_lookup.json --
    # load the actual lookup map
    lm = load_lookup('testing/lemma_lookup.json')
    lemm_lists = [[lm.get(tok, tok) for tok in toks] for toks in token_lists]
    # verify lemmas for each document
    assert lemm_lists == [
        ["i", "be", "bloat", "and", "coughng"],
        ["gas", "and", "burp"],
        ["acid", "reflux", "and", "nausea"],
    ]
    lemm_lists = [
        ["i", "be", "bloat", "and", "coughng"],
        ["gas", "and", "burp"],
        ["acid", "reflux", "and", "naurea"],
    ]
    # -- Step 4: spellcheck --
    sc._disk_cache.clear()
    checked = [spellcheck_token_list(lst) for lst in lemm_lists]
    # expect  'coughng' and 'naurea' corrected
    assert checked == [
        ["i","be","bloat","and","coughing"],
        ["gas","and","burp"],
        ["acid","reflux","and","nausea"],
    ]
    # verify SPELL was called for each token including misspells
    # we expect calls: 'i','was','bloat','and','coughng', ... etc
    processed = [["i","be","bloat","and","cough"],
            ["gas","and","burp"],
            ["acid","reflux","and","nausea"]]
    # -- Step 5: re-lemmatize spellchecked tokens --
    final = [[lm.get(tok, tok) for tok in lst] for lst in checked]

    assert final == processed  # no further changes
