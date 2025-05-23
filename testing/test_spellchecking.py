"""
Spell-check Reddit documents, list every unigram the checker would
“fix”, expand common internet slang before tokenization, spread the heavy work
across CPU cores, and cache results to disk.

Generates in the **current working directory**:
  • changed_ngrams.tsv   – tab-separated list of corrections
  • spell_cache.db       – persistent token→correction cache
"""

import json, sys, re, logging, shelve, atexit, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import random 

# ─────────────────── project imports ────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.append("../vocal_disorder")  # adjust if needed
from spellchecker import SpellChecker
from query_mongo import return_documents

# ───────────────────────────── logging ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # console only
)

# ────────────────── slang expansion helpers ─────────────────────────────────
def load_slang_map(path: str) -> dict[str, str]:
    """
    Reads lines like 'AFAIK=As Far As I Know' and returns a map
    abbr.lower() -> phrase.lower().
    """
    slang = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        abbr, phrase = line.split("=", 1)
        slang[abbr.strip().lower()] = phrase.strip().lower()
    return slang

def expand_slang(text: str, slang_map: dict[str, str]) -> str:
    """
    Replace standalone abbreviations with their full phrase.
    """
    if not slang_map:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in slang_map.keys()) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda m: slang_map[m.group(0).lower()], text)

# ───────────────────── 1. load vocabularies ────────────────────────────────
with open("rcpd_terms.json", encoding="utf-8") as _f:
    TERM_CATEGORY_DICT = json.load(_f)

with open("vocabulary_evaluation/manual_terms.txt", encoding="utf-8") as _f:
    MANUAL_TERMS = [t.strip() for line in _f for t in line.split(",") if t.strip()]

with open("testing/custom_terms.txt", encoding="utf-8") as _f:
    CUSTOM_TERMS = [t.strip() for line in _f for t in line.split(",") if t.strip()]

custom_tokens: set[str] = set()
for terms in TERM_CATEGORY_DICT.values():
    for term in terms:
        custom_tokens.update(term.replace("_", " ").lower().split())
for term in MANUAL_TERMS:
    custom_tokens.update(term.replace("_", " ").lower().split())
for term in CUSTOM_TERMS:
    custom_tokens.update(term.replace("_", " ").lower().split())

# ───────────────────── 2. cache (shared by all runs) ───────────────────────
CACHE_PATH = str(SCRIPT_DIR / "spell_cache.db")
_disk_cache = shelve.open(CACHE_PATH)
atexit.register(_disk_cache.close)

# ───────────────────── 3. multiprocessing helpers ──────────────────────────
def _init_worker(domain_tokens: list[str]):
    global _SPELL
    _SPELL = SpellChecker()
    _SPELL.word_frequency.load_words(domain_tokens)

def _worker(tok: str) -> tuple[str, str]:
    corr = _SPELL.correction(tok) or tok
    return tok, corr

# ───────────────────── 4. fetch documents ──────────────────────────────────
docs = list(
    return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        min_docs=None,
        mongo_uri="mongodb://localhost:27017/",
    )
)
logging.info("Fetched %d documents", len(docs))

# ──────────────────── 5. expand slang ──────────────────────────────────────
slang_map = load_slang_map("testing/slang.txt")
if slang_map:
    docs = [expand_slang(text, slang_map) for text in docs]
    logging.info("Applied slang expansion for %d abbreviations", len(slang_map))

# ───────────────────── 6. scan for unique tokens ───────────────────────────
tokeniser = re.compile(r"[a-z](?:[a-z']*[a-z])?", re.I)
url_pattern = re.compile(r'(?:https?://|www\.|[A-Za-z0-9\-]+\.(?:com|org|io|be)/)\S+', re.I)
mention_pattern = re.compile(r'\b[ur]/[A-Za-z0-9_-]+\b', re.I)
junk_pattern = re.compile(r'[^A-Za-z\' ]+')

unique_tokens: set[str] = set()
for text in tqdm(docs, desc="Scanning tokens"):
    # 1. Remove URLs
    clean = url_pattern.sub(" ", text)
    # 2. Strip u/ and r/ mentions
    clean = mention_pattern.sub(" ", clean)
    # 3. Collapse other non-letter junk
    clean = junk_pattern.sub(" ", clean)
    # 4. Tokenize
    unique_tokens.update(tokeniser.findall(clean.lower()))

logging.info("Unique tokens: %d", len(unique_tokens))
random.seed(42)
for idx, doc in enumerate(random.sample(docs, min(10, len(docs))), start=1):
    # apply the same cleaning pipeline you use above
    clean = url_pattern.sub(" ", doc)
    clean = mention_pattern.sub(" ", clean)
    clean = junk_pattern.sub(" ", clean)
    tokens = tokeniser.findall(clean.lower())
    print(f"\nSample #{idx} tokens:\n{tokens}")
sys.exit(0)

# ───────────────────── 7. parallel spell-check (new tokens only) ───────────
todo = [tok for tok in unique_tokens if tok not in _disk_cache]
if todo:
    n_proc = min(max(mp.cpu_count() - 1, 2), 24)
    logging.info("Spell-checking %d tokens on %d processes…", len(todo), n_proc)
    with mp.Pool(
        processes=n_proc,
        initializer=_init_worker,
        initargs=(list(custom_tokens),),
    ) as pool:
        for orig, corr in tqdm(pool.imap_unordered(_worker, todo, chunksize=500),
                               total=len(todo),
                               desc="Spell-checking unique tokens"):
            _disk_cache[orig] = corr
else:
    logging.info("All tokens already cached – skipping expensive step.")

# ───────────────────── 8. build changed n-grams with cached results ────────
changed: dict[str, str] = {}
for text in tqdm(docs, desc="Building changed unigrams"):
    tokens = tokeniser.findall(text.lower())

    # unigrams only
    for tok in tokens:
        corr = _disk_cache.get(tok, tok)
        if corr != tok:
            changed[tok] = corr

# ───────────────────── 9. write results ────────────────────────────────────
with open(SCRIPT_DIR / "changed_ngrams.tsv", "w", encoding="utf-8") as f:
    for orig, corr in sorted(changed.items()):
        f.write(f"{orig}\t{corr}\n")

logging.info("Wrote %d changed n-grams to changed_ngrams.tsv", len(changed))
print(f"✅ Done. Cache now holds {len(_disk_cache):,} tokens.")
