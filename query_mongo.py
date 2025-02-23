from pymongo import MongoClient
import html
import random

def get_posts_by_subreddit(subreddit_name, db_name="reddit", collection_name="noburp_posts", mongo_uri="mongodb://localhost:27017/"):
    """
    Fetches all posts from a MongoDB collection where the subreddit field matches subreddit_name.

    :param subreddit_name: Name of the subreddit to filter posts.
    :param db_name: Name of the database.
    :param collection_name: Name of the collection.
    :param mongo_uri: MongoDB connection URI.
    :return: List of matching posts.
    """
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Query to filter posts by subreddit
        query = {"subreddit": subreddit_name}
        posts = list(collection.find(query))

        return posts  # Returns a list of documents

    except Exception as e:
        print(f"Error: {e}")
        return []

    finally:
        client.close()  # Close the connection

def get_posts_by_subreddits(subreddit_list, db_name="reddit", collection_name="noburp_posts", mongo_uri="mongodb://localhost:27017/"):
    """
    Fetches all posts from a MongoDB collection where the subreddit field matches any subreddit in subreddit_list.

    :param subreddit_list: List of subreddit names to filter posts.
    :param db_name: Name of the database.
    :param collection_name: Name of the collection.
    :param mongo_uri: MongoDB connection URI.
    :return: List of matching posts.
    """
    if not subreddit_list:
        print("Error: subreddit_list cannot be empty.")
        return []

    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Query to filter posts by multiple subreddits
        query = {"subreddit": {"$in": subreddit_list}}
        posts = list(collection.find(query))

        return posts  # Returns a list of documents

    except Exception as e:
        print(f"Error: {e}")
        return []

    finally:
        client.close()  # Close the connection

# Example usage
if __name__ == "__main__":
    posts = get_posts_by_subreddit("noburp")
    print(f"Found {len(posts)} posts.")
    # Print 5 random posts (for preview)
    random_posts = random.sample(posts, min(5, len(posts)))
    for post in random_posts:
        print(html.unescape(post["selftext"]))
