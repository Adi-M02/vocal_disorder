#!/usr/bin/env python3
"""
nearest_terms_repl.py

Load a Gensim Word2Vec/KeyedVectors model ONCE, then interactively query
nearest neighbors from the CLI. You can change min-sim and topn live.

Usage:
  python nearest_terms_repl.py --model /path/to/model.kv [--min-sim 0.5] [--topn 25] [--lower]

Examples (after it starts):
  term> burp
  term> burp 0.7          # ad-hoc query with custom min-sim
  term> :min 0.65         # persistently set min-sim
  term> :topn 50
  term> :help
  term> :exit

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
    # Format like: term [0.812], term2 [0.701], ...
    return ", ".join(f"{w} [{sim:.3f}]" for w, sim in pairs)


def query_neighbors(kv: KeyedVectors, term: str, min_sim: float, topn: int) -> List[Tuple[str, float]]:
    raw = kv.most_similar(positive=[term], topn=topn)
    return [(w, float(sim)) for (w, sim) in raw if sim >= min_sim]


HELP_TEXT = """\
Commands:
  :help                 Show this help
  :min <float>          Set persistent min-sim threshold (e.g., :min 0.65)
  :topn <int>           Set persistent top-N returned neighbors (e.g., :topn 50)
  :info                 Show current settings
  :exit / :quit / :q    Exit

Queries:
  Just type a term to query neighbors with current settings:
    term> burp

  Or provide an ad-hoc min-sim for a single query:
    term> burp 0.7

Tips:
  • For multi-word tokens, type the exact token used in your vocab (e.g., New_York or no_burp).
  • Use --lower on startup if your vocab is lowercase and you want input auto-lowered.
"""


def main():
    ap = argparse.ArgumentParser(
        description="Interactive nearest-neighbors CLI for a Gensim model."
    )
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to Gensim model (.model/.kv/.bin/.vec/.txt).")
    ap.add_argument("--min-sim", type=float, default=0.4,
                    help="Initial minimum cosine similarity threshold (default: 0.4).")
    ap.add_argument("--topn", type=int, default=25,
                    help="Initial number of neighbors to retrieve before filtering (default: 25).")
    ap.add_argument("--lower", action="store_true", default=True,
                    help="Lowercase each typed term before querying.")
    args = ap.parse_args()

    # Load vectors once
    kv = load_keyed_vectors(args.model)
    vocab_size = len(kv.key_to_index)

    # State
    min_sim = args.min_sim
    topn = args.topn
    vocab_keys = list(kv.key_to_index.keys())

    print(f"\nLoaded model: {args.model}")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Current settings → min_sim={min_sim:.3f}, topn={topn}")
    print("Type :help for commands. Query by typing a term. Ctrl-C/Ctrl-D to exit.\n")

    # REPL loop
    try:
        while True:
            prompt = f"term [min={min_sim:.3f}, topn={topn}]> "
            try:
                line = input(prompt).strip()
            except EOFError:
                print()  # newline on Ctrl-D
                break

            if not line:
                continue

            # Commands
            if line.startswith(":"):
                parts = line.split()
                cmd = parts[0].lower()
                if cmd in (":exit", ":quit", ":q"):
                    break
                elif cmd == ":help":
                    print(HELP_TEXT)
                elif cmd == ":info":
                    print(f"min_sim={min_sim:.3f}, topn={topn}, vocab_size={vocab_size:,}")
                elif cmd == ":min":
                    if len(parts) != 2:
                        print("Usage: :min <float>")
                        continue
                    try:
                        new_min = float(parts[1])
                        if not (-1.0 <= new_min <= 1.0):
                            raise ValueError
                        min_sim = new_min
                        print(f"min_sim set to {min_sim:.3f}")
                    except ValueError:
                        print("Error: min must be a float in [-1.0, 1.0].")
                elif cmd == ":topn":
                    if len(parts) != 2:
                        print("Usage: :topn <int>")
                        continue
                    try:
                        new_topn = int(parts[1])
                        if new_topn <= 0:
                            raise ValueError
                        topn = new_topn
                        print(f"topn set to {topn}")
                    except ValueError:
                        print("Error: topn must be a positive integer.")
                else:
                    print("Unknown command. Type :help for available commands.")
                continue

            # Query path
            # Allow: "<term>" or "<term> <adhoc_min>"
            parts = line.split()
            term = parts[0]
            adhoc_min = None
            if len(parts) >= 2:
                try:
                    adhoc_min = float(parts[-1])
                except ValueError:
                    adhoc_min = None

            if args.lower:
                term = term.lower()

            if term not in kv.key_to_index:
                print(f"✗ '{term}' not in vocabulary.")
                # quick hints: prefix suggestions (up to 10)
                prefix = term[: min(4, len(term))]
                if prefix:
                    candidates = [w for w in vocab_keys if w.startswith(prefix)]
                    if candidates:
                        hint = ", ".join(candidates[:10])
                        print(f"  Suggestions (prefix '{prefix}'): {hint}")
                continue

            eff_min = adhoc_min if adhoc_min is not None else min_sim
            if not (-1.0 <= eff_min <= 1.0):
                print("Error: min-sim must be within [-1.0, 1.0].")
                continue

            try:
                neighbors = query_neighbors(kv, term, eff_min, topn)
            except KeyError:
                print(f"Error: term '{term}' not found in the model.")
                continue

            header = (f"Nearest to '{term}' (min_sim={eff_min:.3f}, topn={topn}):")
            print(header)
            if not neighbors:
                print("  (No neighbors meet the similarity threshold.)")
            else:
                print("  " + format_neighbors(neighbors))

    except KeyboardInterrupt:
        print("\nInterrupted.")

if __name__ == "__main__":
    main()
