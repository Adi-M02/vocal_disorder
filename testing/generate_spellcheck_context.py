import json
import sys
import logging
import shelve
import atexit
import multiprocessing as mp
from pathlib import Path
from collections import Counter

from tqdm import tqdm
from spellchecker import SpellChecker

# ───────────────────────────────────────────────────────────────────────────────
# Project imports (adjust path if needed)
SCRIPT_DIR = Path(__file__).parent
sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# ───────────────────────── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ────────────────────── 1. Load vocabularies ───────────────────────────────────
with open("rcpd_terms.json", encoding="utf-8") as f:
    TERM_CATEGORY_DICT = json.load(f)

with open("testing/custom_terms.txt", encoding="utf-8") as f:
    CUSTOM_TERMS = [t.strip() for line in f for t in line.split(",") if t.strip()]

# Build a set of domain-specific tokens (split on whitespace, lowercase)
custom_tokens: set[str] = set()
for terms in TERM_CATEGORY_DICT.values():
    for term in terms:
        custom_tokens.update(term.replace("_", " ").lower().split())
for term in CUSTOM_TERMS:
    custom_tokens.update(term.replace("_", " ").lower().split())

# ───────────────────── 2. Initialize cache ───────────────────────────────────────
CACHE_DIR = SCRIPT_DIR / "spell_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PREFIX = str(CACHE_DIR / "spell_cache")
_disk_cache = shelve.open(CACHE_PREFIX)
atexit.register(_disk_cache.close)

# ─────────────────── 3. Multiprocessing spell-check helpers ─────────────────────
def _init_worker(domain_tokens: list[str]):
    """
    Initialize a SpellChecker in each worker, loading domain-specific words.
    """
    global _SPELL
    _SPELL = SpellChecker()
    _SPELL.word_frequency.load_words(domain_tokens)
    # Ensure digit-containing tokens remain “known” and are not corrected:
    for tok in domain_tokens:
        if any(ch.isdigit() for ch in tok):
            _SPELL.word_frequency.add(tok)

def _worker(tok: str) -> tuple[str, str]:
    """
    Spell-check a single token. If it contains a digit, skip correction.
    """
    if any(ch.isdigit() for ch in tok):
        return tok, tok

    corr = _SPELL.correction(tok) or tok
    return tok, corr

# ─────────────────── 4. Fetch documents ─────────────────────────────────────────
def fetch_documents():
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        min_docs=None,
        mongo_uri="mongodb://localhost:27017/",
    )
    logging.info("Fetched %d documents", len(docs))
    return docs

# ─────────────────── Main processing ────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="For tokens corrected more than --min_count times, "
                    "extract up to --examples_per_term contexts and write them "
                    "in descending‐frequency order to a TXT file."
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=100,
        help="Only consider tokens that were corrected more than this many times."
    )
    parser.add_argument(
        "--examples_per_term",
        type=int,
        default=3,
        help="How many example contexts to extract per high-frequency token."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to write the term→contexts TXT output."
    )
    args = parser.parse_args()

    # ─────────────────── 5. (Optional) Slang-expansion could go here… ──────────────

    # ─────────────────── 6. Scan for unique tokens ─────────────────────────────────
    docs_raw = fetch_documents()
    unique_tokens: set[str] = set()
    tokenized_docs: list[list[str]] = []

    logging.info("Tokenizing all documents…")
    for text in tqdm(docs_raw, desc="Tokenizing"):
        tokens = clean_and_tokenize(text)
        tokenized_docs.append(tokens)
        unique_tokens.update(tokens)

    logging.info("Unique tokens extracted: %d", len(unique_tokens))

    # ─────────────────── 7. Parallel spell-check ───────────────────────────────────
    todo = [tok for tok in unique_tokens if tok not in _disk_cache]
    if todo:
        n_proc = min(mp.cpu_count() - 1, 24)
        logging.info("Spell-checking %d tokens on %d processes…", len(todo), n_proc)

        with mp.Pool(
            processes=n_proc,
            initializer=_init_worker,
            initargs=(list(custom_tokens),),
        ) as pool:
            for orig, corr in tqdm(
                pool.imap_unordered(_worker, todo, chunksize=500),
                total=len(todo),
                desc="Spell-checking unique tokens"
            ):
                _disk_cache[orig] = corr
    else:
        logging.info("All tokens already cached – skipping spell-check.")

    # ─────────────────── 8. Build changed-unigram COUNTS ───────────────────────────
    counts = Counter()
    for tokens in tqdm(tokenized_docs, desc="Counting changed tokens"):
        for tok in tokens:
            corr = _disk_cache.get(tok, tok)
            if corr != tok:
                counts[tok] += 1

    logging.info("Found %d distinct corrected tokens", len(counts))

    # ─────────────────── 9. Filter tokens by frequency threshold ───────────────────
    freq_tokens = {tok: cnt for tok, cnt in counts.items() if cnt > args.min_count}
    logging.info(
        "%d tokens corrected > %d times",
        len(freq_tokens),
        args.min_count
    )

    if not freq_tokens:
        logging.info("No tokens exceed the threshold; exiting.")
        return

    # ─────────────────── 10. Extract contexts for each frequent token ─────────────
    # We’ll take up to `examples_per_term` windows of ±5 tokens around each occurrence.
    window_radius = 5
    contexts_per_term: dict[str, list[str]] = {tok: [] for tok in freq_tokens}

    logging.info("Extracting up to %d contexts per token…", args.examples_per_term)
    for tokens in tqdm(tokenized_docs, desc="Scanning docs for contexts"):
        for i, tok in enumerate(tokens):
            if tok in freq_tokens and len(contexts_per_term[tok]) < args.examples_per_term:
                start = max(0, i - window_radius)
                end = min(len(tokens), i + window_radius + 1)
                window = tokens[start:end]
                snippet = " ".join(window)
                contexts_per_term[tok].append(snippet)
        # Early exit if we’ve collected enough for all tokens:
        if all(len(contexts_per_term[tok]) >= args.examples_per_term for tok in contexts_per_term):
            break

    # ─────────────────── 11. Write output in descending‐frequency TXT format ──────
    out_path = args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort tokens by descending count
    sorted_tokens = sorted(freq_tokens.items(), key=lambda x: x[1], reverse=True)
    # sorted_tokens is a list of (token, count) pairs

    with out_path.open("w", encoding="utf-8") as fout:
        for token, cnt in sorted_tokens:
            # Print the original token
            fout.write(f"term: {token}\n")
            # Print how many times it was corrected
            fout.write(f"frequency: {cnt}\n")
            # Print what it was corrected to
            corrected = _disk_cache.get(token, token)
            fout.write(f"corrected_to: {corrected}\n")

            # Print example contexts
            snippets = contexts_per_term.get(token, [])
            for idx, snip in enumerate(snippets, start=1):
                fout.write(f"context {idx}: {snip}\n")
            fout.write("\n")  # blank line between tokens

    logging.info(
        "Wrote contexts for %d tokens (threshold > %d) to %s",
        len(sorted_tokens),
        args.min_count,
        out_path
    )

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()