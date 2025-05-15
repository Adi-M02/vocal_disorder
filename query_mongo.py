import pymongo
from pymongo import MongoClient
from datetime import datetime
from collections import defaultdict
import pandas as pd
import random
from pathlib import Path
import logging
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

def get_text_by_subreddits(subreddit_list, db_name="reddit", collection_name="noburp_all", mongo_uri="mongodb://localhost:27017/"):
    """
    Fetches readable post texts from a MongoDB collection filtered by subreddits.

    :param subreddit_list: List of subreddit names.
    :param db_name: MongoDB database name.
    :param collection_name: Collection name.
    :param mongo_uri: MongoDB URI.
    :return: List of strings (combined title/selftext or body).
    """

    if not subreddit_list:
        print("Error: subreddit_list cannot be empty.")
        return []

    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        query = {"subreddit": {"$in": subreddit_list}}
        texts = []
        for entry in collection.find(query):
            # Safely extract fields
            title = html.unescape(entry.get("title", "") or "")
            selftext = html.unescape(entry.get("selftext", "") or "")
            body = html.unescape(entry.get("body", "") or "")

            # Prefer post-style content
            combined = f"{title} {selftext}".strip() if selftext else body.strip()
            if combined and combined.lower() not in {"[deleted]", "[removed]"}:
                texts.append(combined)

        return texts

    except Exception as e:
        print(f"Error: {e}")
        return []

    finally:
        client.close()

def get_posts_by_users(users, subreddits=None, db_name="reddit", collection_name="noburp_all", mongo_uri="mongodb://localhost:27017/"):
    """
    Returns a list of posts from the specified users, filtered by the given subreddits if provided.
    
    Parameters:
        users (list): List of user names (strings) whose posts to retrieve.
        subreddits (list, optional): List of subreddit names (strings) to filter posts.
                                     If None or empty, returns all posts by the users.
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection containing posts.
        mongo_uri (str): MongoDB connection string.
    
    Returns:
        list: A list of post documents matching the query.
    """
    # Build the query to include only posts by the specified users.
    query = {"author": {"$in": users}}
    # If a list of subreddits is provided, add it to the query.
    if subreddits and len(subreddits) > 0:
        query["subreddit"] = {"$in": subreddits}
    
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    posts = list(collection.find(query))
    client.close()
    
    return posts


def write_all_users_posts(
    db_name, collection_name, output_file,
    filter_subreddits=None, filter_num_posts=None, filter_user=None, 
    mongo_uri="mongodb://localhost:27017/"
):
    import pymongo
    import html
    import logging
    from datetime import datetime
    from collections import defaultdict

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Connecting to MongoDB at %s", mongo_uri)

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Step 1: Query posts
    if filter_user:
        logging.info("Fetching posts only for user: %s", filter_user)
        posts_cursor = collection.find({"author": filter_user})
    else:
        logging.info("Fetching all posts from collection '%s'", collection_name)
        posts_cursor = collection.find({})

    # Step 2: Group posts by author
    posts_by_author = defaultdict(list)
    total_posts = 0
    for post in posts_cursor:
        author = post.get("author", "Unknown")
        posts_by_author[author].append(post)
        total_posts += 1

    logging.info("Total posts retrieved: %d", total_posts)
    logging.info("Total authors found: %d", len(posts_by_author))

    # Step 3: Sort authors
    if filter_user:
        sorted_authors = [filter_user] if filter_user in posts_by_author else []
    else:
        if filter_subreddits:
            filter_list = [s.lower() for s in filter_subreddits]
            sorted_authors = sorted(
                posts_by_author.keys(),
                key=lambda a: sum(1 for post in posts_by_author[a] if post.get("subreddit", "").lower() in filter_list),
                reverse=True
            )
        else:
            sorted_authors = sorted(posts_by_author.keys(), key=lambda a: len(posts_by_author[a]), reverse=True)

    # Step 4: Write to output file
    total_written_posts = 0
    logging.info("Writing output to file: %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, author in enumerate(sorted_authors, start=1):
            author_posts = posts_by_author[author]
            total_author_posts = len(author_posts)

            # Filter by subreddits if requested
            if filter_subreddits:
                filter_list = [s.lower() for s in filter_subreddits]
                filtered_posts = [post for post in author_posts if post.get("subreddit", "").lower() in filter_list]
            else:
                filtered_posts = author_posts

            count_filtered = len(filtered_posts)

            # Skip if not enough posts
            if filter_num_posts and count_filtered <= filter_num_posts:
                continue

            f.write("#" * 80 + "\n")
            header = f"Author: {author} - Total posts: {total_author_posts} - Filtered posts: {count_filtered}"
            f.write(header + "\n")
            f.write("#" * 80 + "\n")

            logging.info("Processing author %d/%d: %s with %d filtered posts", i, len(sorted_authors), author, count_filtered)

            # Sort posts chronologically
            filtered_posts.sort(key=lambda p: p.get("created_utc", 0))

            posts_written_for_author = 0
            for post in filtered_posts:
                subreddit = post.get("subreddit", "N/A")
                timestamp = post.get("created_utc", None)
                if isinstance(timestamp, (int, float)):
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = "Unknown time"

                title = post.get("title", "N/A")
                selftext = html.unescape(post.get("selftext", "N/A"))
                body = post.get("body", None)

                f.write("=" * 80 + "\n")
                f.write(f"Subreddit: {subreddit}\n")
                f.write(f"Time     : {time_str}\n")
                if body:
                    f.write("body :\n")
                    f.write(body + "\n")
                else:
                    f.write(f"Title    : {title}\n")
                    f.write("Selftext :\n")
                    f.write(selftext + "\n")
                f.write("=" * 80 + "\n\n")

                posts_written_for_author += 1
                total_written_posts += 1

            f.write("\n\n")
            logging.info("Finished writing %d posts for author: %s", posts_written_for_author, author)

    client.close()
    logging.info("Finished writing all authors' posts. Total posts written: %d", total_written_posts)
    logging.info("Output saved to %s", output_file)

def write_selected_users_posts(
    db_name, collection_name, output_file,
    filter_subreddits=None, filter_num_posts=None, filter_users=None, 
    mongo_uri="mongodb://localhost:27017/"
):
    import pymongo
    import html
    import logging
    from datetime import datetime
    from collections import defaultdict

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Connecting to MongoDB at %s", mongo_uri)

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Step 1: Query posts
    query = {}
    if filter_users:
        logging.info("Fetching posts only for %d users", len(filter_users))
        query["author"] = {"$in": filter_users}
    else:
        logging.info("Fetching all posts from collection '%s'", collection_name)

    posts_cursor = collection.find(query)

    # Step 2: Group posts by author
    posts_by_author = defaultdict(list)
    total_posts = 0
    for post in posts_cursor:
        author = post.get("author", "Unknown")
        posts_by_author[author].append(post)
        total_posts += 1

    logging.info("Total posts retrieved: %d", total_posts)
    logging.info("Total authors found: %d", len(posts_by_author))

    # Step 3: Sort authors
    if filter_users:
        sorted_authors = [u for u in filter_users if u in posts_by_author]
    else:
        if filter_subreddits:
            filter_list = [s.lower() for s in filter_subreddits]
            sorted_authors = sorted(
                posts_by_author.keys(),
                key=lambda a: sum(1 for post in posts_by_author[a] if post.get("subreddit", "").lower() in filter_list),
                reverse=True
            )
        else:
            sorted_authors = sorted(posts_by_author.keys(), key=lambda a: len(posts_by_author[a]), reverse=True)

    # Step 4: Write to output file
    total_written_posts = 0
    logging.info("Writing output to file: %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, author in enumerate(sorted_authors, start=1):
            author_posts = posts_by_author[author]
            total_author_posts = len(author_posts)

            # Filter by subreddits if requested
            if filter_subreddits:
                filter_list = [s.lower() for s in filter_subreddits]
                filtered_posts = [post for post in author_posts if post.get("subreddit", "").lower() in filter_list]
            else:
                filtered_posts = author_posts

            count_filtered = len(filtered_posts)

            # Skip if not enough posts
            if filter_num_posts and count_filtered <= filter_num_posts:
                continue

            f.write("#" * 80 + "\n")
            header = f"Author: {author} - Total posts: {total_author_posts} - Filtered posts: {count_filtered}"
            f.write(header + "\n")
            f.write("#" * 80 + "\n")

            logging.info("Processing author %d/%d: %s with %d filtered posts", i, len(sorted_authors), author, count_filtered)

            # Sort posts chronologically
            filtered_posts.sort(key=lambda p: p.get("created_utc", 0))

            posts_written_for_author = 0
            for post in filtered_posts:
                subreddit = post.get("subreddit", "N/A")
                timestamp = post.get("created_utc", None)
                if isinstance(timestamp, (int, float)):
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = "Unknown time"

                title = post.get("title", "N/A")
                selftext = html.unescape(post.get("selftext", "N/A"))
                body = post.get("body", None)

                f.write("=" * 80 + "\n")
                f.write(f"Subreddit: {subreddit}\n")
                f.write(f"Time     : {time_str}\n")
                if body:
                    f.write("body :\n")
                    f.write(body + "\n")
                else:
                    f.write(f"Title    : {title}\n")
                    f.write("Selftext :\n")
                    f.write(selftext + "\n")
                f.write("=" * 80 + "\n\n")

                posts_written_for_author += 1
                total_written_posts += 1

            f.write("\n\n")
            logging.info("Finished writing %d posts for author: %s", posts_written_for_author, author)

    client.close()
    logging.info("Finished writing all selected authors' posts. Total posts written: %d", total_written_posts)
    logging.info("Output saved to %s", output_file)

def generate_post_samples_and_write_output(
    other_users_csv,
    botox_users_csv,
    db_name,
    collection_name,
    output_path_batch1,
    output_path_batch2,
    filter_subreddits=None,
    filter_num_posts=None,
    mongo_uri="mongodb://localhost:27017/",
    seed=42
):

    # === Load CSVs ===
    other_users_df = pd.read_csv(other_users_csv)
    botox_users_df = pd.read_csv(botox_users_csv)

    other_users = set(other_users_df['user'].dropna().unique())
    botox_users = set(botox_users_df['user'].dropna().unique())

    # Ensure reproducibility
    random.seed(seed)

    # === Sample 25 from each list for batch 1 ===
    other_sample1 = set(random.sample(other_users, 25))
    botox_sample1 = set(random.sample(botox_users, 25))

    # Remove batch 1 users to avoid overlap in batch 2
    remaining_self_cured = other_users - other_sample1
    remaining_botox = botox_users - botox_sample1

    # === Sample 25 from each list for batch 2 ===
    other_sample2 = set(random.sample(remaining_self_cured, 25))
    botox_sample2 = set(random.sample(remaining_botox, 25))

    batch1_users = list(other_sample1 | botox_sample1)
    batch2_users = list(other_sample2 | botox_sample2)

    # === Write each batch to output ===
    write_selected_users_posts(
        db_name=db_name,
        collection_name=collection_name,
        output_file=output_path_batch1,
        filter_subreddits=filter_subreddits,
        filter_num_posts=3,
        filter_users=batch1_users,
        mongo_uri=mongo_uri
    )

    write_selected_users_posts(
        db_name=db_name,
        collection_name=collection_name,
        output_file=output_path_batch2,
        filter_subreddits=filter_subreddits,
        filter_num_posts=3,
        filter_users=batch2_users,
        mongo_uri=mongo_uri
    )

def return_user_entries(
    db_name,
    collection_name,
    user=None,
    filter_subreddits=None,
    mongo_uri="mongodb://localhost:27017/"
):
    if not user:
        raise ValueError("You must specify a user.")

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Build query for the user and optional subreddit filter
    query = {"author": user}
    if filter_subreddits:
        filter_list = [s.lower() for s in filter_subreddits]
        query["subreddit"] = {"$in": filter_list}

    # Run the query
    user_posts = list(collection.find(query))
    client.close()

    return user_posts

def return_multiple_users_entries(
    db_name,
    collection_name,
    users,
    filter_subreddits=None,
    mongo_uri="mongodb://localhost:27017/"
):
    if not users or not isinstance(users, list):
        raise ValueError("You must pass a non-empty list of users.")

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Build query for multiple users and optional subreddit filter
    query = {"author": {"$in": users}}
    if filter_subreddits:
        filter_list = [s.lower() for s in filter_subreddits]
        query["subreddit"] = {"$in": filter_list}

    # Run the query
    all_posts = list(collection.find(query))
    client.close()

    return all_posts

    


if __name__ == "__main__":
    # filter_subreddits = ["noburp", "emetophobia", "anxiety", "gerd", "ibs", "sibo", "emetophobiarecovery", "pots", "gastritis", "healthanxiety", "trees", "advice", "supplements"]
    filter_subreddits = ["noburp"]
    # write_all_users_posts("reddit", "noburp_all", "vocabulary_evaluation/mjh59.txt", filter_subreddits, None, "mjh59" )
    other_users_csv = "other_users.csv"
    botox_csv = "user_botox_dates_fixed.csv"
    output_path_batch1 = "output_annotation_batches/user_posts_batch1.txt"
    output_path_batch2 = "output_annotation_batches/user_posts_batch2.txt"
    generate_post_samples_and_write_output(
        other_users_csv,
        botox_csv,
        db_name="reddit",
        collection_name="noburp_all",
        output_path_batch1=output_path_batch1,
        output_path_batch2=output_path_batch2,
        filter_subreddits=filter_subreddits,
        filter_num_posts=3,
        mongo_uri="mongodb://localhost:27017/"
    )