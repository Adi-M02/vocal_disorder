# test mongo pipeline on a fake mongomock database
# pytest unit_testing/test_retrieval.py -s 
import sys
import pytest
import mongomock

sys.path.append('../vocal_disorder')
import query_mongo
from query_mongo import return_documents

@pytest.fixture(autouse=True)
def mock_mongo_client(monkeypatch):
    # This is the one client your code will see:
    client = mongomock.MongoClient()
    monkeypatch.setattr(
        query_mongo.pymongo,
        "MongoClient",
        lambda uri: client
    )
    return client

def test_comments_and_posts(mock_mongo_client):
    # Use the fixture’s client, not a new one
    db = mock_mongo_client["reddit"]
    coll = db["noburp_all"]
    coll.insert_many([
        {"body": "just a comment", "subreddit": "noburp", "author": "u1"},
        {"title": "Post Title", "selftext": "post body", "subreddit": "noburp", "author": "u2"},
        {"body": " ", "title": "", "selftext": " ", "subreddit": "noburp", "author": "u3"},
    ])

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    print(docs)  # For debugging
    assert docs == ["just a comment", "Post Title", "post body"]


def test_no_filters_behavior(mock_mongo_client):
    db = mock_mongo_client["reddit"]
    coll = db["noburp_all"]
    coll.insert_many([
        {"body": "body1", "subreddit": "noburp", "author": "a"},
        {"title": "title2", "selftext": "text2", "subreddit": "noburp", "author": "b"},
        {"title": "", "selftext": "", "subreddit": "noburp", "author": "c"},
    ])

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        mongo_uri="mongodb://localhost:27017/",
    )
    assert docs == ["body1", "title2", "text2"]


def test_filter_users_only(mock_mongo_client):
    db = mock_mongo_client["reddit"]
    coll = db["noburp_all"]
    coll.insert_many([
        {"body": "keep me", "subreddit": "noburp", "author": "keepme"},
        {"body": "drop me", "subreddit": "noburp", "author": "other"},
    ])

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=[],
        filter_users=["keepme"],
        mongo_uri="mongodb://localhost:27017/",
    )
    assert docs == ["keep me"]


def test_min_docs_filter(mock_mongo_client):
    db = mock_mongo_client["reddit"]
    coll = db["noburp_all"]
    coll.insert_many([
        {"body": "a1", "subreddit": "noburp", "author": "alice"},
        {"body": "a2", "subreddit": "noburp", "author": "alice"},
        {"body": "b1", "subreddit": "noburp", "author": "bob"},
    ])

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        min_docs=2,
        mongo_uri="mongodb://localhost:27017/",
    )
    assert docs == ["a1", "a2"]


def test_min_docs_and_user_intersection(mock_mongo_client):
    db = mock_mongo_client["reddit"]
    coll = db["noburp_all"]
    coll.insert_many([
        {"body": "a1", "subreddit": "noburp", "author": "alice"},
        {"body": "a2", "subreddit": "noburp", "author": "alice"},
        {"body": "b1", "subreddit": "noburp", "author": "bob"},
    ])

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        min_docs=2,
        filter_users=["nobody"],
        mongo_uri="mongodb://localhost:27017/",
    )
    assert docs == []


def test_subreddit_filter_applied(mock_mongo_client):
    db = mock_mongo_client["reddit"]
    coll = db["noburp_all"]
    coll.insert_many([
        {"body": "in foo", "subreddit": "foo", "author": "u1"},
        {"body": "in noburp", "subreddit": "noburp", "author": "u2"},
    ])

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    assert docs == ["in noburp"]
