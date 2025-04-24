import pymongo
import pandas as pd
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import addcopyfighandler
from datetime import datetime, timedelta

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


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_botox_dates(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    date_cols = [c for c in df.columns if "botox" in c.lower() and "date" in c.lower()]
    if not date_cols:
        raise KeyError("No botox-date column found in CSV.")
    col = date_cols[0]
    df[col] = pd.to_datetime(df[col], format="%m-%d-%y", errors="coerce")
    return dict(zip(df["user"].astype(str), df[col]))

def fetch_noburp_posts(db_name: str,
                       coll_name: str,
                       mongo_uri: str = "mongodb://localhost:27017/") -> pd.DataFrame:
    client = pymongo.MongoClient(mongo_uri)
    coll   = client[db_name][coll_name]
    docs   = list(coll.find({}, {"author":1, "created_utc":1}))
    client.close()

    # build DataFrame
    df = pd.DataFrame(docs)
    df = df.rename(columns={"created_utc":"ts_unix"})
    df["ts"] = pd.to_datetime(df["ts_unix"], unit="s")
    df = df[["author","ts"]].dropna()
    
    logging.info("Fetched %d posts for %d unique users",
                 len(df), df["author"].nunique())
    return df

def compute_rate_counts(df: pd.DataFrame,
                        botox_dates: dict,
                        windows: list[tuple[str,int,int]]) -> pd.DataFrame:
    records = []
    # for each user, slice df once
    users = list(botox_dates.keys())
    logging.info("Computing rates for %d users …", len(users))
    for user in users:
        botox_dt = botox_dates[user]
        if pd.isna(botox_dt):
            continue
        user_posts = df[df["author"] == user]
        for label, start_off, end_off in windows:
            start = botox_dt + timedelta(days=start_off)
            end   = botox_dt + timedelta(days=end_off)
            cnt   = user_posts[(user_posts["ts"] >= start) & (user_posts["ts"] <= end)].shape[0]
            days  = max((end-start).days, 1)
            rate  = cnt / days
            records.append((user, label, cnt, days, rate))
    return pd.DataFrame.from_records(
        records,
        columns=["user","window","count","days","posts_per_day"]
    ).set_index(["user","window"])

def plot_rate_summary(rate_df: pd.DataFrame, windows: list):
    order = [w[0] for w in windows]
    agg   = rate_df.groupby("window")["posts_per_day"].agg(["mean","std"]).reindex(order)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(agg.index, agg["mean"], yerr=agg["std"], capsize=4)
    ax.set_xlabel("Window")
    ax.set_ylabel("Posts per Day")
    ax.set_title("Posting Rate Before vs. After Botox")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    CSV_FILE   = "user_botox_dates_fixed.csv"
    DB_NAME    = "reddit"
    COLL_NAME  = "noburp_posts"
    botox_dates = load_botox_dates(CSV_FILE)

    # 1) Fetch once
    posts_df = fetch_noburp_posts(DB_NAME, COLL_NAME)

    # 2) Define windows
    windows = [
        ("baseline", -30, -1),
        ("early",      1, 10),
        ("mid",       11, 30),
        ("long",     31,180),
    ]

    # 3) Compute in‑memory
    rate_df = compute_rate_counts(posts_df, botox_dates, windows)

    # 4) Show summary
    summary = (
        rate_df.reset_index()
               .groupby("window")["posts_per_day"]
               .agg(["mean","std","count"])
               .reindex([w[0] for w in windows])
    )
    print(summary.to_string(float_format="{:6.3f}".format))

    # 5) Plot
    plot_rate_summary(rate_df, windows)