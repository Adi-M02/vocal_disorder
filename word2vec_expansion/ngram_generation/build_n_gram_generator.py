"""
build n grams with tfidf to maximize coverage of seed terms
"""

import sys
sys.path.append('../vocal_disorder')
from utils.load_json import load_json
from utils.text_pipeline import process_text
from utils.load_and_process_docs import process_all_noburp

