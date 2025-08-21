#!/usr/bin/env python3
"""
check_vocab.py — interactive vocabulary checker for Gensim Word2Vec/KeyedVectors.

Usage
-----
# Interactive REPL (type terms; Enter on empty line or 'q' to quit)
python check_vocab.py --model path/to/model.model

# One-off term
python check_vocab.py --model path/to/model.model --term "able_to_burp"

# File of terms (one per line; '#' comments ignored)
python check_vocab.py --model path/to/model.model --file terms.txt

# Disable/enable normalization heuristics
python check_vocab.py --model path/to/model.model --no-lower --no-underscore
"""

from __future__ import annotations
import argparse
import sys
import unicodedata as ud
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

def load_w2v_any(path: Path):
    """Load a Gensim Word2Vec model or KeyedVectors from common formats."""
    try:
        from gensim.models import Word2Vec, KeyedVectors
    except Exception as e:
        raise SystemExit("This script requires gensim to be installed: pip install gensim") from e

    ext = path.suffix.lower()
    # 1) Full Word2Vec model
    if ext in {".model", ".w2v"}:
        try:
            m = Word2Vec.load(str(path))
            return m.wv
        except Exception:
            pass
    # 2) KeyedVectors saved via .save
    try:
        kv = KeyedVectors.load(str(path), mmap='r')
        return kv
    except Exception:
        pass
    # 3) word2vec format (binary or text)
    if ext in {".bin", ".txt", ".vec"}:
        from gensim.models import KeyedVectors as KV
        try:
            binary = ext == ".bin"
            kv = KV.load_word2vec_format(str(path), binary=binary)
            return kv
        except Exception:
            pass
    raise SystemExit(f"Unrecognized or unsupported model file: {path}")

def has_index(kv, token: str) -> bool:
    """Membership check compatible with Gensim 4.x."""
    try:
        return kv.has_index_for(token)
    except Exception:
        return token in kv  # fallback

def norm_variants(
    token: str,
    try_nfkc: bool = True,
    try_lower: bool = True,
    try_underscore: bool = True
) -> List[Tuple[str, str]]:
    """
    Produce candidate variants for a token with labels describing the transform.
    Order preserves increasing aggressiveness.
    """
    variants: List[Tuple[str, str]] = []
    seen = set()

    def add(v: str, label: str):
        vv = v.strip()
        if vv not in seen:
            seen.add(vv)
            variants.append((vv, label))

    # raw & stripped
    add(token, "raw")
    add(token.strip(), "strip")

    # NFKC normalization
    if try_nfkc:
        add(ud.normalize('NFKC', token), "nfkc")
        add(ud.normalize('NFKC', token.strip()), "strip+nfkc")

    # lowercase
    if try_lower:
        add(token.lower(), "lower")
        if try_nfkc:
            add(ud.normalize('NFKC', token).lower(), "nfkc+lower")
            add(ud.normalize('NFKC', token.strip()).lower(), "strip+nfkc+lower")

    # space→underscore
    if try_underscore:
        add(token.replace(" ", "_"), "space→underscore")
        if try_lower:
            add(token.replace(" ", "_").lower(), "space→underscore+lower")
        if try_nfkc:
            add(ud.normalize('NFKC', token).replace(" ", "_"), "nfkc+space→underscore")
            if try_lower:
                add(ud.normalize('NFKC', token).replace(" ", "_").lower(), "nfkc+space→underscore+lower")

    return variants

def check_one(kv, term: str, try_nfkc=True, try_lower=True, try_underscore=True) -> Dict[str, Any]:
    """Return a dict with membership details and which variant matched."""
    variants = norm_variants(term, try_nfkc=try_nfkc, try_lower=try_lower, try_underscore=try_underscore)
    for v, label in variants:
        if v and has_index(kv, v):
            return {
                "term": term,
                "in_vocab": True,
                "matched_variant": v,
                "transform": label
            }
    return {
        "term": term,
        "in_vocab": False,
        "matched_variant": None,
        "transform": None
    }

def iter_terms_from_file(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield s

def main():
    ap = argparse.ArgumentParser(description="Check whether terms exist in a Word2Vec/KeyedVectors vocabulary.")
    ap.add_argument("--model", required=True, help="Path to Gensim model/KeyedVectors (.model/.bin/.txt/.vec)")
    ap.add_argument("--term", help="Single term to check (non-interactive)")
    ap.add_argument("--file", help="Path to a file with one term per line")
    ap.add_argument("--no-nfkc", dest="nfkc", action="store_false", help="Disable NFKC normalization attempts")
    ap.add_argument("--no-lower", dest="lower", action="store_false", help="Disable lowercase attempts")
    ap.add_argument("--no-underscore", dest="underscore", action="store_false", help="Disable space→underscore attempts")
    args = ap.parse_args()

    model_path = Path(args.model)
    kv = load_w2v_any(model_path)
    print(f"[info] Loaded vectors: {len(kv.key_to_index):,} tokens | dim={kv.vector_size} from {model_path}")

    def handle(term: str):
        res = check_one(kv, term, try_nfkc=args.nfkc, try_lower=args.lower, try_underscore=args.underscore)
        if res["in_vocab"]:
            print(f"✔ IN-VOCAB: {res['term']}  → matched='{res['matched_variant']}' ({res['transform']})")
        else:
            print(f"✘ OOV: {res['term']}")

    # Non-interactive modes
    if args.term:
        handle(args.term)
        return
    if args.file:
        for t in iter_terms_from_file(Path(args.file)):
            handle(t)
        return

    # Interactive REPL
    print("\nType a term to check (Enter on empty line or 'q' to quit).")
    try:
        while True:
            term = input("> ").strip()
            if term == "" or term.lower() in {"q", "quit", "exit"}:
                break
            handle(term)
    except (KeyboardInterrupt, EOFError):
        print("\n[bye]")

if __name__ == "__main__":
    main()
