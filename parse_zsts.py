import zstandard
import logging
import os
import json
import csv
import logging.handlers
from collections import defaultdict
from pymongo import MongoClient
from pymongo.errors import PyMongoError


log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

OUR_set = set() 
def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    
    if not chunk:  # End of file reached
        return ""

    bytes_read += len(chunk)

    if previous_chunk:
        chunk = previous_chunk + chunk

    try:
        return chunk.decode()
    except UnicodeDecodeError:
        log.warning(f"Decoding error at {bytes_read:,} bytes, skipping problematic characters.")
        
        # Option 1: Try decoding with 'replace' to avoid breaking
        return chunk.decode(errors="replace")
      
def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()

def get_field_values(input_folder):
    field_types = defaultdict(lambda: defaultdict(set))
    inputfilenumber = 0
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if not filename.startswith("."):
            filepath = os.path.join(input_folder, filename)
            log.info(f"{filename}, {inputfilenumber}")
            inputfilenumber += 1
            i = 0

            # Process each line in the file
            for line, file_bytes_processed in read_lines_zst(filepath):
                try:
                    obj = json.loads(line)
                    # Analyze only the first 10 JSONs in each file
                    if i < 10:
                        # Loop through each key-value pair in the JSON object
                        for key, value in obj.items():
                            value_type = type(value).__name__
                            if value_type != "NoneType":  # Skip None values
                                field_types[key][value_type].add(repr(value))
                        i += 1
                    else:
                        break
                except json.JSONDecodeError:
                    log.error(f"Error decoding JSON in file {filename} at line {i+1}")
    for field, types in field_types.items():
        if len(types) > 1:
            print(f"Field '{field}' has multiple types:")
            for value_type, examples in types.items():
                example = next(iter(examples))  # Get an example value
                print(f"  Type '{value_type}': Example value {example}")

def get_comment_field_values(input_folder):
    for filename in os.listdir(input_folder):
        if not filename.startswith("."):
            filepath = os.path.join(input_folder, filename)
            i = 0
            field_types = defaultdict(lambda: defaultdict(set))

            # Process each line in the file
            for line, file_bytes_processed in read_lines_zst(filepath):
                try:
                    obj = json.loads(line)
                    
                    # Analyze only the first 10 JSONs in each file
                    if i < 10:
                        log.info(f"Analyzing entry {i+1} from file {filename}")

                        # Loop through each key-value pair in the JSON object
                        for key, value in obj.items():
                            value_type = type(value).__name__
                            if value_type != "NoneType":  # Skip None values
                                field_types[key][value_type].add(repr(value))

                        i += 1
                    else:
                        break
                except json.JSONDecodeError:
                    log.error(f"Error decoding JSON in file {filename} at line {i+1}")

    # Printing field types and examples
    for field, types in field_types.items():
        if len(types) > 1:
            print(f"Field '{field}' has multiple types:")
        else:
            print(f"Field '{field}' has a single type:")
        
        for value_type, examples in types.items():
            example = next(iter(examples))  # Get an example value
            print(f"  Type '{value_type}': Example value {example}")

def process_json_object(obj):
    # Drop the 'edited' field if it exists
    if "edited" in obj:
        del obj["edited"]
    # Cast 'created' and 'created_utc' to int if they exist
    if "created" in obj:
        try:
            obj["created"] = int(obj["created"])
        except (ValueError, TypeError):
            log.warning(f"Failed to cast 'created' to int.")
    if "created_utc" in obj:
        try:
            obj["created_utc"] = int(obj["created_utc"])
        except (ValueError, TypeError):
            log.warning(f"Failed to cast 'created_utc' to int.")
    if "subreddit" in obj:
        obj["subreddit"] = obj["subreddit"].lower()
    return obj

def enter_posts_to_collection(input_folder, collection, author_set):
    inputfilenumber = 1
    batch = []
    batch_size = 1000

    for filename in os.listdir(input_folder):
        if not filename.startswith("."):
            filepath = os.path.join(input_folder, filename)
            log.info(f"Processing {filename}, file number {inputfilenumber}")
            inputfilenumber += 1
            i = 0
            for line, file_bytes_processed in read_lines_zst(filepath):
                try:
                    obj = json.loads(line)
                    if obj.get('author', '') in author_set:
                        obj = process_json_object(obj)
                        obj['is_post'] = True
                        if obj['selftext'] not in ["", "[removed]"]:
                            batch.append(obj)
                    i += 1
                    if len(batch) >= batch_size:
                        try:
                            collection.insert_many(batch, ordered=False)
                        except PyMongoError as err:
                            log.info(f"Batch insertion error: {err}")
                        batch = []
                    if i % 500000 == 0:
                        log.info(f"Processed {i} lines from {filename}")
                except json.JSONDecodeError as err:
                    log.info(f"JSON decode error: {err}")
                except Exception as err:
                    log.info(f"Unexpected error: {err}")
            # Insert any remaining documents in the batch
            if batch:
                try:
                    collection.insert_many(batch, ordered=False)
                    log.info(f"Successfully inserted the final batch of {len(batch)} documents for {filename}.")
                except PyMongoError as err:
                    log.info(f"Final batch insertion error: {err}")
            batch = []

def enter_comments_to_collection(input_folder, collection):
    inputfilenumber = 1
    batch = []
    batch_size = 1000

    for filename in os.listdir(input_folder):
        if not filename.startswith("."):
            filepath = os.path.join(input_folder, filename)
            log.info(f"Processing {filename}, file number {inputfilenumber}")
            inputfilenumber += 1
            i = 0
            for line, file_bytes_processed in read_lines_zst(filepath):
                try:
                    obj = json.loads(line)
                    if obj.get('author', '') in OUR_set:
                        obj = process_json_object(obj)
                        obj['is_post'] = False
                        if obj['body'] not in ["", "[removed]"]:
                            batch.append(obj)
                    i += 1
                    if len(batch) >= batch_size:
                        try:
                            collection.insert_many(batch, ordered=False)
                        except PyMongoError as err:
                            log.info(f"Batch insertion error: {err}")
                        batch = []
                    if i % 500000 == 0:
                        log.info(f"Processed {i} lines from {filename}")
                except json.JSONDecodeError as err:
                    log.info(f"JSON decode error: {err}")
                except Exception as err:
                    log.info(f"Unexpected error: {err}")
            # Insert any remaining documents in the batch
            if batch:
                try:
                    collection.insert_many(batch, ordered=False)
                    log.info(f"Successfully inserted the final batch of {len(batch)} documents for {filename}.")
                except PyMongoError as err:
                    log.info(f"Final batch insertion error: {err}")
            batch = []

if __name__ == "__main__":
    client = MongoClient()
    db = client['reddit']
    post_collection = db['posts_test']
    comment_collection = db['comments_test']
    posts_folder = "/local/disk1/not_backed_up/amukundan/2009_to_2022_reddit_zsts/reddit/submissions/"
    comments_folder = "/local/disk2/not_backed_up/amukundan/reddit_comments/reddit/09_to_22_comments/"
    # enter_posts_to_collection(posts_folder, post_collection)
    # enter_comments_to_collection(comments_folder, comment_collection)
    get_comment_field_values(comments_folder) 