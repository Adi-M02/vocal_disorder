#!/usr/bin/env python3
import os
import json
import argparse
import sys
from gensim.models import Word2Vec

# allow importing your project’s tokenizer
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize


def load_terms(path):
    """
    Load category→list-of-terms from JSON, replacing underscores with spaces.
    """
    with open(path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    # replace underscores in categories and terms
    return {
        category.replace('_', ' '): [term.replace('_', ' ') for term in terms]
        for category, terms in terms_map.items()
    }


def component_status(model: Word2Vec, comp: str) -> bool:
    """
    Tokenize the component and return True if all tokens are in model's vocab.
    """
    toks = clean_and_tokenize(comp)
    return bool(toks) and all(tok in model.wv.key_to_index for tok in toks)


def main():
    parser = argparse.ArgumentParser(
        description="Break underscore-terms into components and check vocab membership"
    )
    parser.add_argument(
        '--model_dir', required=True,
        help="Folder containing word2vec_cbow.model & word2vec_skipgram.model"
    )
    parser.add_argument(
        '--terms', default='rcpd_terms.json',
        help="Path to your rcpd_terms.json"
    )
    args = parser.parse_args()

    # locate models
    cbow_path = os.path.join(args.model_dir, 'word2vec_cbow.model')
    skip_path = os.path.join(args.model_dir, 'word2vec_skipgram.model')
    for path in (cbow_path, skip_path, args.terms):
        if not os.path.exists(path):
            parser.error(f"Not found: {path}")

    # load models
    print(f"Loading CBOW model from {cbow_path}…")
    cbow = Word2Vec.load(cbow_path)
    print(f"Loading Skip-gram model from {skip_path}…")
    skip = Word2Vec.load(skip_path)

    # load terms with underscores replaced
    terms_map = load_terms(args.terms)

    # iterate categories and terms
    for category, terms in terms_map.items():
        print(f"\n=== Category: {category} ===")
        for term in terms:
            comps = term.split(' ')
            print(f"\nTerm: {term}")
            for comp in comps:
                in_cbow = component_status(cbow, comp)
                in_skip = component_status(skip, comp)
                print(
                    f"  {comp:15}  CBOW: {'Yes' if in_cbow else ' No'}    "
                    f"Skip-gram: {'Yes' if in_skip else ' No'}"
                )

if __name__ == '__main__':
    main()
