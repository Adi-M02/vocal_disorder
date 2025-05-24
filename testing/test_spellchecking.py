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
from collections import Counter
import unicodedata

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
CACHE_DIR = SCRIPT_DIR / "spell_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PREFIX = str(CACHE_DIR / "spell_cache")  
_disk_cache   = shelve.open(CACHE_PREFIX)
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

# ───────────────────── 6. scan for unique tokens # allow internal apostrophes and hyphens, preserve subreddit/user slash
tokeniser   = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'/\-]*[A-Za-z0-9])?", re.I)
url_pattern = re.compile(r'(?:https?://|www\.|[A-Za-z0-9\-]+\.(?:com|org|io|be)/)\S+', re.I)
# remove everything except letters, numbers, apostrophe, hyphen, slash, or space
junk_pattern = re.compile(r"[^A-Za-z0-9'/\- ]+")


def clean_and_tokenize(text: str) -> list[str]:
    """
    Apply URL removal, slash preservation for r/ and u/, junk stripping, then extract tokens.
    """
    clean = unicodedata.normalize("NFKC", text)
    clean = clean.replace("–", "-").replace("—", "-")
    clean = clean.replace("‘", "'").replace("’", "'")
    clean = clean.replace("…", "...")
    clean = clean.replace("\u200B", "").replace("\u00A0", " ")
    # 1) strip URLs
    clean = url_pattern.sub(" ", clean)
    # 2) preserve subreddit/user slashes (r/xxx or u/xxx)
    clean = re.sub(r"\b([ru])/", r"\1<SLASH>", clean, flags=re.I)
    # 3) replace any other slash with space
    clean = clean.replace("/", " ")
    # 4) restore preserved slash
    clean = clean.replace("<SLASH>", "/")
    # 5) drop other junk but keep letters, apostrophes, hyphens, and spaces
    clean = junk_pattern.sub(" ", clean)
    # 6) tokenize on lowercase
    return tokeniser.findall(clean.lower())

unique_tokens: set[str] = set()
for text in tqdm(docs, desc="Scanning tokens"):
    # use the full clean/token pipeline you tested above
    tokens = clean_and_tokenize(text)
    unique_tokens.update(tokens)

logging.info("Unique tokens: %d", len(unique_tokens))
# random.seed(42)
# for idx, doc in enumerate(random.sample(docs, min(10, len(docs))), start=1):
#     # apply the same cleaning pipeline you use above
#     clean = url_pattern.sub(" ", doc)
#     clean = mention_pattern.sub(" ", clean)
#     clean = junk_pattern.sub(" ", clean)
#     tokens = tokeniser.findall(clean.lower())
#     print(f"\nSample #{idx} tokens:\n{tokens}")
# sys.exit(0)

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

# ───────────────────── 8. build changed unigrams WITH COUNTS ─────────────────
counts = Counter()
for text in tqdm(docs, desc="Counting changed unigrams"):
    tokens = tokeniser.findall(text.lower())
    for tok in tokens:
        corr = _disk_cache.get(tok, tok)
        if corr != tok:
            counts[(tok, corr)] += 1

# ───────────────────── 9. write results SORTED BY COUNT ────────────────────
with open(str(SCRIPT_DIR / "changed_ngrams.tsv" ), "w", encoding="utf-8") as f:
    # header
    f.write("original\tcorrected\tcount\n")
    # most_common() yields in descending order by count
    for (orig, corr), cnt in counts.most_common():
        f.write(f"{orig}\t{corr}\t{cnt}\n")
