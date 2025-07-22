"""splits manual terms into a seed term json and a evaluation terms txt file
usage: python word2vec_expansion/split_manual_terms.py --manual_terms <path> --outdir <path> --split <float>"""
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list

def load_lookup(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

lookup = load_lookup("testing/lemma_lookup.json")

def tok_fn(text: str) -> List[str]:
    tokens = clean_and_tokenize(text)
    tokens = [lookup.get(t, t) for t in tokens]
    tokens = spellcheck_token_list(tokens)
    tokens = [lookup.get(t, t) for t in tokens]
    return tokens

