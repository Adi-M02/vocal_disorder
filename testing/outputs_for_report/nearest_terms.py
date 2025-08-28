#!/usr/bin/env python3
"""
nearest_terms.py

Load a Gensim Word2Vec/KeyedVectors model and print the top-N nearest
terms to an input term, filtered by a minimum cosine similarity.

Examples:
  python nearest_terms.py --model /path/to/model.kv --term "burp" \
      --topn 25 --min-sim 0.6

Supported model files (auto-detected by extension):
  - .model / .model.gz         (gensim Word2Vec models)
  - .kv / .kv.gz               (gensim KeyedVectors)
  - .bin                       (word2vec binary format)
  - .vec / .txt                (word2vec text format)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple, List

try:
    from gensim.models import Word2Vec, KeyedVectors
except Exception as e:
    print("Error: gensim is required. Try: pip install gensim", file=sys.stderr)
    raise

EXT_WORD2VEC = {".model", ".model.gz"}
EXT_KEYEDVEC = {".kv", ".kv.gz"}
EXT_W2V_BIN  = {".bin"}
EXT_W2V_TXT  = {".vec", ".txt"}


def detect_format(path: Path) -> str:
    p = str(path).lower()
    for ext in EXT_WORD2VEC:
        if p.endswith(ext):
            return "w2v_model"
    for ext in EXT_KEYEDVEC:
        if p.endswith(ext):
            return "keyed"
    for ext in EXT_W2V_BIN:
        if p.endswith(ext):
            return "w2v_bin"
    for ext in EXT_W2V_TXT:
        if p.endswith(ext):
            return "w2v_txt"
    # Fallback: try KeyedVectors.load, then Word2Vec.load, then word2vec_format
    return "auto"


def load_keyed_vectors(path: Path) -> KeyedVectors:
    """Load various Gensim/word2vec formats and return a KeyedVectors object."""
    kind = detect_format(path)

    if kind == "w2v_model":
        model: Word2Vec = Word2Vec.load(str(path))
        return model.wv
    elif kind == "keyed":
        return KeyedVectors.load(str(path), mmap="r")
    elif kind == "w2v_bin":
        return KeyedVectors.load_word2vec_format(str(path), binary=True)
    elif kind == "w2v_txt":
        return KeyedVectors.load_word2vec_format(str(path), binary=False)
    else:
        # Try a few strategies in order
        # 1) KeyedVectors.load
        try:
            return KeyedVectors.load(str(path), mmap="r")
        except Exception:
            pass
        # 2) Word2Vec.load
        try:
            model = Word2Vec.load(str(path))
            return model.wv
        except Exception:
            pass
        # 3) Assume word2vec binary, then text
        try:
            return KeyedVectors.load_word2vec_format(str(path), binary=True)
        except Exception:
            try:
                return KeyedVectors.load_word2vec_format(str(path), binary=False)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load vectors from {path}. "
                    "Tried KeyedVectors, Word2Vec, word2vec binary, and word2vec text."
                ) from e


def format_neighbors(pairs: List[Tuple[str, float]]) -> str:
    # Format like: term (0.812), term2 (0.701), ...
    return ", ".join(f"{w} [{sim:.3f}]" for w, sim in pairs)


def main():
    ap = argparse.ArgumentParser(
        description="Find nearest terms in a Gensim model by cosine similarity."
    )
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to Gensim model (.model/.kv/.bin/.vec/.txt).")
    ap.add_argument("--term", required=True,
                    help="Query term that must exist in the model vocabulary.")
    ap.add_argument("--min-sim", type=float, default=0.4,
                    help="Minimum cosine similarity threshold (default: 0.5).")
    args = ap.parse_args()

    # Load vectors
    kv = load_keyed_vectors(args.model)

    # Quick vocab check
    if args.term not in kv.key_to_index:
        print(
            f"Error: term '{args.term}' is not in the model vocabulary "
            f"(vocab size: {len(kv.key_to_index)}).",
            file=sys.stderr,
        )
        sys.exit(2)


    try:
        raw = kv.most_similar(positive=[args.term], topn=25)
    except KeyError:
        # Shouldn't happen because we checked membership, but just in case
        print(f"Error: term '{args.term}' not found in the model.", file=sys.stderr)
        sys.exit(2)

    # Filter by threshold and clip to topn
    filtered = [(w, float(sim)) for (w, sim) in raw if sim >= args.min_sim]
    topk = filtered[:25]

    header = (f"Nearest to '{args.term}' "
              f"(min_sim={args.min_sim}, topn={25}):")
    if not topk:
        print(header)
        print("No neighbors meet the similarity threshold.")
        sys.exit(0)

    print(header)
    print(format_neighbors(topk))


if __name__ == "__main__":
    main()
