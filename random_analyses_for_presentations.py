import pymongo
import pandas as pd
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

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


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # … assume calculate_non_botox_percentages is already defined above …
    db_name          = "reddit"
    collection_name  = "noburp_posts"
    botox_csv_file   = "user_botox_dates_fixed.csv"
    target_subreddit = "noburp"

    # 1) compute
    stats = calculate_non_botox_percentages(
        db_name,
        collection_name,
        botox_csv_file,
        target_subreddit=target_subreddit
    )

    # 2) build DataFrame
    df = pd.DataFrame.from_dict(
        stats,
        orient="index",
        columns=["total_users", "non_botox_users", "pct_non_botox"]
    )
    df.index.name = "num_posts"

    # 3) reindex for exactly 10→4, introducing NaNs if missing
    desired = list(range(10, 3, -1))   # [10, 9, …, 4]
    df = df.reindex(desired)

    # 4) drop any rows where we never had users
    df = df.dropna(how="all")

    # 5) print nicely
    print("\nDetailed table (10→4):")
    print(
        df.to_string(formatters={
            "total_users":     "{:>5}".format,
            "non_botox_users": "{:>5}".format,
            "pct_non_botox":   "{:6.2f}%".format
        })
    )

    # 6) plot in that exact order
    fig, ax = plt.subplots()
    ax.bar(df.index.astype(int), df["pct_non_botox"])
    ax.set_xlabel("Number of Posts")
    ax.set_ylabel("Percentage of Non‑Botox Users")
    ax.set_title(f"Non‑Botox % by Post Count in r/{target_subreddit}")

    # ensure ticks are exactly 10→4
    ax.set_xticks(desired)
    # if you still want the *visual* left‑to‑right to be 10→4
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

