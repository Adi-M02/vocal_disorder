import pymongo
import pandas as pd
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import addcopyfighandler

def calculate_non_botox_percentages(
    db_name: str,
    collection_name: str,
    botox_csv_file: str,
    target_subreddit: str = None,
    mongo_uri: str = "mongodb://localhost:27017/"
) -> dict:
    """
    Connects to the given MongoDB collection, counts the number of posts per author
    (optionally only in `target_subreddit`), loads the CSV of known botox users,
    and for each post-count N from 20 down to 4 prints and returns the percentage
    of users with N posts who have never had botox.
    """

    # set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Loading botox user list from %s", botox_csv_file)

    # 1) Load CSV of botox users
    botox_df = pd.read_csv(botox_csv_file)
    if "user" in botox_df.columns:
        botox_users = set(botox_df["user"].dropna().astype(str))
    else:
        botox_users = set(botox_df.iloc[:, 0].dropna().astype(str))
    logging.info("Found %d unique botox users", len(botox_users))

    # 2) Connect to MongoDB
    logging.info("Connecting to MongoDB at %s", mongo_uri)
    client = pymongo.MongoClient(mongo_uri)
    coll = client[db_name][collection_name]

    # 3) Build query filter
    query_filter = {}
    if target_subreddit:
        query_filter["subreddit"] = target_subreddit
        logging.info("Filtering posts to subreddit: %s", target_subreddit)

    # 4) Count posts per author (in the filter)
    post_counts = defaultdict(int)
    for doc in coll.find(query_filter, {"author": 1}):
        author = doc.get("author")
        if author:
            post_counts[author] += 1
    logging.info("Computed post counts for %d authors", len(post_counts))

    # 5) Group authors by post‐count
    counts_group = defaultdict(list)
    for author, cnt in post_counts.items():
        counts_group[cnt].append(author)

    # 6) Compute percentages for N = 20 … 4
    results = {}
    header = f"--- Stats for subreddit '{target_subreddit}' ---" if target_subreddit else "--- Stats for all subreddits ---"
    print(header)
    for N in range(20, 3, -1):
        authors_n = counts_group.get(N, [])
        total_n = len(authors_n)
        if total_n == 0:
            continue

        non_botox_n = sum(1 for a in authors_n if a not in botox_users)
        pct_non_botox = non_botox_n / total_n * 100
        print(f"{N:2d} posts: {non_botox_n:3d}/{total_n:3d} non‑botox users ({pct_non_botox:5.2f}%)")

        results[N] = {
            "total_users": total_n,
            "non_botox_users": non_botox_n,
            "pct_non_botox": pct_non_botox
        }

    client.close()
    return results

def avg_wordcount_by_postcount(
    db_name: str,
    collection_name: str,
    target_subreddit: str,
    post_min: int = 4,
    post_max: int = 10,
    mongo_uri: str = "mongodb://localhost:27017/"
) -> dict:
    """
    For each author in `target_subreddit`, count how many posts they have
    and how many total words across title+selftext. Then for each N in
    post_max..post_min, compute average word‐count per user (only users with
    exactly N posts).
    Returns: { N: {"num_users": int, "avg_words": float} }
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Connecting to MongoDB at %s", mongo_uri)
    client = pymongo.MongoClient(mongo_uri)
    coll = client[db_name][collection_name]

    # accumulate per-user stats
    stats = defaultdict(lambda: {"count": 0, "words": 0})
    logging.info("Scanning posts in r/%s …", target_subreddit)
    for doc in coll.find({"subreddit": target_subreddit}, {"author":1, "title":1, "selftext":1}):
        author = doc.get("author")
        if not author:
            continue
        stats[author]["count"] += 1

        # combine title + selftext, split on whitespace
        text = ""
        if doc.get("title"):
            text += doc["title"] + " "
        if doc.get("selftext"):
            text += doc["selftext"]
        # simple word count
        stats[author]["words"] += len(text.split())

    client.close()
    logging.info("Finished scanning %d users", len(stats))

    # group by post-count
    group = defaultdict(list)
    for author, vals in stats.items():
        cnt = vals["count"]
        group[cnt].append(vals["words"])

    # for each N in [post_max..post_min], compute average
    results = {}
    for N in range(post_max, post_min - 1, -1):
        word_lists = group.get(N, [])
        num_users = len(word_lists)
        if num_users == 0:
            continue
        avg_words = sum(word_lists) / num_users
        results[N] = {"num_users": num_users, "avg_words": avg_words}

    return results


if __name__ == "__main__":
    db_name          = "reddit"
    collection_name  = "noburp_posts"
    target_subreddit = "noburp"

    # 1) compute
    wc_stats = avg_wordcount_by_postcount(
        db_name,
        collection_name,
        target_subreddit,
        post_min=4,
        post_max=10
    )

    # 2) build DataFrame
    df = pd.DataFrame.from_dict(
        wc_stats, orient="index", columns=["num_users", "avg_words"]
    )
    df.index.name = "num_posts"
    df = df.sort_index(ascending=False)

    # 3) print table
    print("\nAverage Word‑Count per User (10→4 posts):")
    print(df.to_string(formatters={
        "num_users": "{:>5}".format,
        "avg_words": "{:8.1f}".format
    }))

    # 4) plot
    fig, ax = plt.subplots()
    ax.bar(df.index.astype(int), df["avg_words"])
    ax.set_xlabel("Number of Posts in r/noburp")
    ax.set_ylabel("Avg. Total Words per User")
    ax.set_title("Avg. Word‑Count by Post Count (10→4 posts)")
    ax.set_xticks(df.index.tolist())
    ax.invert_xaxis()      # so it runs left→right: 10,9,…,4
    plt.tight_layout()
    plt.show()
