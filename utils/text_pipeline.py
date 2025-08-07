import sys
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize
from utils.load_lemmatizer import load_lookup
from spellchecker_folder.spellchecker import spellcheck_token_list


def process_text(text, lookup_path='testing/combined_lemmas.json'):
    "process a dcoument. tokenize -> lemmatize -> retokenize -> spellcheck -> relemmatize -> retokenize"""
    lookup_map = load_lookup(lookup_path)

    toks = clean_and_tokenize(text)
    toks = [lookup_map.get(tok, tok) for tok in toks]
    text = ' '.join(toks)
    toks = clean_and_tokenize(text)
    toks = spellcheck_token_list(toks)
    toks = [lookup_map.get(tok, tok) for tok in toks]
    text = ' '.join(toks)
    toks = clean_and_tokenize(text)
    return toks