import os 
import json
import multiprocessing
from multiprocessing import Value, Manager
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import logging
import parse_zsts as pz

AUTHOR_SET = set()
with open("user_lists/no_burp_users_2.txt", "r") as f:
     for line in f:
         AUTHOR_SET.add(line.strip())

BATCH_SIZE = 1000



def get_authors(args):
    file_path, subreddit, file_number, total_files = args
    user_set = set()
    
    print(f"Processing file {file_number}/{total_files}: {file_path}")

    try:
        for line, _ in pz.read_lines_zst(file_path):
            try:
                obj = json.loads(line)
                if obj.get("subreddit", "").lower() == subreddit.lower():
                    author = obj.get("author")
                    if author and author not in {"[deleted]", "AutoModerator"}:
                        user_set.add(author)
            except json.JSONDecodeError:
                print(f"JSON decode error in file {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return user_set

def get_user_set(input_folder, subreddit, num_workers):
    ignored_files = ['RS_2021-07.zst']
    files = [
        (os.path.join(input_folder, file_name), subreddit, idx + 1, len(os.listdir(input_folder)))
        for idx, file_name in enumerate(os.listdir(input_folder))
        if not file_name.startswith(".") and file_name.endswith(".zst") and file_name not in ignored_files
    ]

    user_set = set()

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(get_authors, files):
            user_set.update(result)  # Merge sets from workers

    return user_set

def process_file(filepath, db_name, collection_name, author_set, progress, total_files):
    """
    Processes a single .zst file and inserts matching posts into MongoDB.
    """
    client = MongoClient(connecttimeoutms=60000, serverselectiontimeoutms=60000)
    db = client[db_name]
    collection = db[collection_name]

    batch = []
    processed_lines = 0

    log.info(f"Starting processing: {filepath}")

    try:
        for line, _ in pz.read_lines_zst(filepath):
            try:
                obj = json.loads(line)
                if obj.get('author') in author_set:
                    obj = pz.process_json_object(obj)
                    batch.append(obj)

                processed_lines += 1
                if len(batch) >= BATCH_SIZE:
                    collection.insert_many(batch, ordered=False)
                    batch = []

                if processed_lines % 500000 == 0:
                    log.info(f"{filepath}: Processed {processed_lines} lines.")

            except json.JSONDecodeError as err:
                log.warning(f"JSON decode error in {filepath}: {err}")
            except Exception as err:
                log.warning(f"Unexpected error in {filepath}: {err}")

        if batch:
            collection.insert_many(batch, ordered=False)
            log.info(f"Inserted final batch of {len(batch)} for {filepath}.")

    except Exception as err:
        log.error(f"Error processing {filepath}: {err}")

    client.close()

    # Track progress using a shared dictionary
    progress[filepath] = True
    completed_files = len(progress)
    log.info(f"Completed {completed_files}/{total_files}: {filepath}")


def enter_posts_to_mongodb(input_folder, db_name, collection_name, author_set):
    """
    Uses multiprocessing to process all .zst files in input_folder.
    """
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".zst")]
    total_files = len(files)
    
    log.info(f"Total files to process: {total_files}")
    
    start_time = time.time()

    with Manager() as manager:
        progress = manager.dict()  # Shared dictionary instead of Value

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_file, file, db_name, collection_name, author_set, progress, total_files): file for file in files}

            for future in tqdm(as_completed(futures), total=total_files, desc="Processing files"):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error processing {futures[future]}: {e}")

        elapsed_time = time.time() - start_time
        log.info(f"Finished processing all {total_files} files in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    log = logging.getLogger("bot")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    input_folder = "/local/disk3/not_backed_up/amukundan/2009_to_2025_posts/"
    subreddit = "noburp"
    # users = get_user_set(input_folder, subreddit, num_workers=os.cpu_count())
    # with open("user_lists/no_burp_users_2.txt", "w") as f:
    #     for user in users:
    #         f.write(user + "\n")
    client = MongoClient()
    db = client["reddit"]
    collection = db["noburp_posts"]
    try:
        client.admin.command('ping')
        log.info("MongoDB connection successful.")
    except Exception as e:
        log.info(f"MongoDB connection failed: {e}")

    enter_posts_to_mongodb(input_folder, "reddit", "noburp_posts_2", AUTHOR_SET)
