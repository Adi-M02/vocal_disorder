import csv
import re
from pymongo import MongoClient
from collections import Counter

def parse_undated_users(input_file):
    """
    Parses users from a file grouped under category headers.
    Returns a list of dictionaries with 'user' and 'source_category' keys.
    """
    undated_users = []
    current_category = None

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if re.match(r".*:$", line):
            current_category = line.rstrip(":").strip()
        elif re.match(r".*?:\s", line):
            parts = line.split(":", 1)
            current_category = parts[0].strip()
            users = [u.strip() for u in parts[1].split(",") if u.strip()]
            for user in users:
                undated_users.append({"user": user, "source_category": current_category})
        elif current_category:
            users = [u.strip() for u in line.split(",") if u.strip()]
            for user in users:
                undated_users.append({"user": user, "source_category": current_category})
    
    return undated_users

def write_users_to_csv(users, output_file):
    """
    Writes a list of user dictionaries to a CSV file.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "source_category"])
        writer.writeheader()
        writer.writerows(users)

    print(f"Wrote {len(users)} undated users to '{output_file}'")

def load_usernames_from_csv(files):
    """
    Loads all usernames from 'user' column in a list of CSV files.
    Returns a set of usernames and a Counter of duplicates.
    """
    usernames = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = row.get("user")
                if username:
                    usernames.append(username.strip())
    return set(usernames), Counter(usernames)

def load_usernames_from_mongo(mongo_uri, db_name, collection_name):
    """
    Loads all distinct 'author' usernames from MongoDB.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    return set(collection.distinct("author"))

def show_missing_and_duplicates(csv_users, mongo_users, csv_counts):
    """
    Prints users in CSV but not in MongoDB, and duplicates in CSVs.
    """
    # Users in CSV but not in MongoDB
    missing_in_mongo = sorted(csv_users - mongo_users)
    print(f"\n❌ Users in CSV but NOT in MongoDB ({len(missing_in_mongo)}):")
    for user in missing_in_mongo:
        print(f"  - {user}")

    # Duplicate users in CSV
    print("\n♻️ Duplicate users in CSV:")
    for user, count in csv_counts.items():
        if count > 1:
            print(f"  - {user}: {count} times")

if __name__ == "__main__":
    csv_files = ["other_users.csv", "user_botox_dates.csv"]
    mongo_uri = "mongodb://localhost:27017"
    db_name = "reddit"
    collection_name = "noburp_posts_2"

    csv_users, csv_counts = load_usernames_from_csv(csv_files)
    mongo_users = load_usernames_from_mongo(mongo_uri, db_name, collection_name)

    show_missing_and_duplicates(csv_users, mongo_users, csv_counts)

