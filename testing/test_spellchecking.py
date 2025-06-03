"""
Spell-check Reddit documents, list every unigram the checker would
“fix”, expand common internet slang before tokenization (optional),
spread the heavy work across CPU cores, and cache results to disk.

Generates in the **current working directory**:
  • changed_ngrams.tsv   – tab-separated list of corrections
  • spell_cache.db       – persistent token→correction cache
"""

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

# Project imports
SCRIPT_DIR = Path(__file__).parent
sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# ───────────────────────── Logging ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ────────────────────── 1. Load vocabularies ─────────────────────────────────
with open("rcpd_terms.json", encoding="utf-8") as f:
    TERM_CATEGORY_DICT = json.load(f)

with open("vocabulary_evaluation/manual_terms.txt", encoding="utf-8") as f:
    MANUAL_TERMS = [t.strip() for line in f for t in line.split(",") if t.strip()]

with open("testing/custom_terms.txt", encoding="utf-8") as f:
    CUSTOM_TERMS = [t.strip() for line in f for t in line.split(",") if t.strip()]

# Build a set of domain-specific tokens (split on whitespace, lowercase)
custom_tokens: set[str] = set()
for terms in TERM_CATEGORY_DICT.values():
    for term in terms:
        custom_tokens.update(term.replace("_", " ").lower().split())
for term in MANUAL_TERMS:
    custom_tokens.update(term.replace("_", " ").lower().split())
for term in CUSTOM_TERMS:
    custom_tokens.update(term.replace("_", " ").lower().split())

# ───────────────────── 2. Initialize cache ────────────────────────────────────
CACHE_DIR = SCRIPT_DIR / "spell_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PREFIX = str(CACHE_DIR / "spell_cache")
_disk_cache = shelve.open(CACHE_PREFIX)
atexit.register(_disk_cache.close)

# ─────────────────── 3. Multiprocessing spell-check helpers ───────────────────
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

# ─────────────────── 4. Fetch documents ───────────────────────────────────────
docs = return_documents(
    db_name="reddit",
    collection_name="noburp_all",
    filter_subreddits=["noburp"],
    min_docs=None,
    mongo_uri="mongodb://localhost:27017/",
)
logging.info("Fetched %d documents", len(docs))

# ─────────────────── 5. (Optional) Expand slang ─────────────────────────────────
# Uncomment and adjust the path to enable slang expansion.
#
# def load_slang_map(path: str) -> dict[str, str]:
#     slang = {}
#     for line in Path(path).read_text(encoding="utf-8").splitlines():
#         if "=" not in line:
#             continue
#         abbr, phrase = line.split("=", 1)
#         slang[abbr.strip().lower()] = phrase.strip().lower()
#     return slang
#
# def expand_slang(text: str, slang_map: dict[str, str]) -> str:
#     pattern = re.compile(
#         r"\b(" + "|".join(re.escape(k) for k in slang_map.keys()) + r")\b",
#         flags=re.IGNORECASE,
#     )
#     return pattern.sub(lambda m: slang_map[m.group(0).lower()], text)
#
# slang_map = load_slang_map("testing/slang.txt")
# if slang_map:
#     docs = [expand_slang(text, slang_map) for text in docs]
#     logging.info("Applied slang expansion for %d abbreviations", len(slang_map))

# ─────────────────── 6. Scan for unique tokens ────────────────────────────────
unique_tokens: set[str] = set()
for text in tqdm(docs, desc="Scanning tokens"):
    tokens = clean_and_tokenize(text)
    unique_tokens.update(tokens)

logging.info("Unique tokens extracted: %d", len(unique_tokens))

# ─────────────────── 7. Parallel spell-check (new tokens only) ────────────────
todo = [tok for tok in unique_tokens if tok not in _disk_cache]
if todo:
    n_proc = min(max(mp.cpu_count() - 1, 2), 24)
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

# ─────────────────── 8. Build changed unigrams WITH COUNTS ─────────────────────
counts = Counter()
for text in tqdm(docs, desc="Counting changed unigrams"):
    tokens = clean_and_tokenize(text)
    for tok in tokens:
        corr = _disk_cache.get(tok, tok)
        if corr != tok:
            counts[(tok, corr)] += 1

# ─────────────────── 9. Write results SORTED BY COUNT ────────────────────────
output_path = SCRIPT_DIR / "changed_ngrams.tsv"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("original\tcorrected\tcount\n")
    for (orig, corr), cnt in counts.most_common():
        f.write(f"{orig}\t{corr}\t{cnt}\n")

logging.info("Finished writing changed n-grams to %s", output_path)
