import sys
import json
import re
import shelve
import atexit
import unicodedata

from pathlib import Path
from spellchecker import SpellChecker
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize
# ─────────────────────────────────────────────────────────────────────────────
# 1) Base directory for this module; cache will live under “spell_cache/”
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# 2) Set up on‐disk cache (shelve). Every time we correct a token, we store it.
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = SCRIPT_DIR / "spell_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PREFIX = str(CACHE_DIR / "spell_cache")

# Open a shelve db; this maps “raw_token” → “corrected_token”
_disk_cache = shelve.open(CACHE_PREFIX)
atexit.register(_disk_cache.close)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Load your TERM_CATEGORY_DICT (rcpd_terms.json) and CUSTOM_TERMS (custom_terms.txt)
# ─────────────────────────────────────────────────────────────────────────────
_with_custom = SCRIPT_DIR / "testing" / "custom_terms.txt"

# Load the JSON mapping from categories → [list of terms]
with open('rcpd_terms_6_5.json', encoding="utf-8") as f:
    TERM_CATEGORY_DICT = json.load(f)

# Load a flat list of custom terms from the txt (comma‐separated)
with open('spellchecker_folder/custom_terms.txt', encoding="utf-8") as f:
    CUSTOM_TERMS = [t.strip() for line in f for t in line.split(",") if t.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# 4) Build a single set of “custom_tokens” by splitting on underscores & spaces
#    so that every individual word from your category‐terms or your custom_terms
#    ends up in this set. 
# ─────────────────────────────────────────────────────────────────────────────
custom_tokens = set()

# From TERM_CATEGORY_DICT:
for terms_list in TERM_CATEGORY_DICT.values():
    for term in terms_list:
        # e.g. if term = "retrograde_cricopharyngeal", we do:
        #    "retrograde cricopharyngeal".lower().split() → ["retrograde", "cricopharyngeal"]
        for tok in term.replace("_", " ").lower().split():
            custom_tokens.add(tok)

# From CUSTOM_TERMS:
for term in CUSTOM_TERMS:
    for tok in term.replace("_", " ").lower().split():
        custom_tokens.add(tok)


# ─────────────────────────────────────────────────────────────────────────────
# 5) Instantiate a SpellChecker that only knows our “custom_tokens” by default.
#    (language=None → blank English base, then we load only our words.)
# ─────────────────────────────────────────────────────────────────────────────
SPELL = SpellChecker(language=None)
SPELL.word_frequency.load_words(custom_tokens)



# ─────────────────────────────────────────────────────────────────────────────
# 6) Precompile all patterns needed for clean/tokenize
# ─────────────────────────────────────────────────────────────────────────────
_tokeniser = re.compile(
    r"[A-Za-z\u00C0-\u024F0-9](?:[A-Za-z\u00C0-\u024F0-9'/\-]*[A-Za-z\u00C0-\u024F0-9])?",
    re.I
)
_url_pattern = re.compile(
    r'(?:https?://|www\.|[A-Za-z0-9\-]+\.(?:com|org|io|be)/)\S+', 
    re.I
)
_junk_pattern = re.compile(r"[^A-Za-z0-9'/\- ]+")



# ─────────────────────────────────────────────────────────────────────────────
# 7) Exactly your original clean_and_tokenize (no changes)
# ────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 8) Helper: correct a single token (using cache + SpellChecker + digit‐skip)
# ─────────────────────────────────────────────────────────────────────────────
def _correct_token(token: str) -> str:
    """
    If token in custom_tokens → return as-is.
    Else if token in _disk_cache → return cached correction.
    Else if token contains a digit → skip correction, cache it → return token.
    Otherwise, ask SPELL.correction(token), cache that, return it.
    """
    # 1) If it’s one of our known “custom” words, no correction needed
    if token in custom_tokens:
        return token

    # 2) If we’ve already corrected this token before, grab it from disk
    if token in _disk_cache:
        return _disk_cache[token]

    # 3) If it has any digit (e.g. “h2o”, “2025”), skip correction
    if any(ch.isdigit() for ch in token):
        _disk_cache[token] = token
        return token

    # 4) Otherwise, ask SpellChecker for its best guess
    suggestion = SPELL.correction(token)
    if suggestion is None:
        suggestion = token

    # 5) Store in disk cache and return
    _disk_cache[token] = suggestion
    return suggestion



# ─────────────────────────────────────────────────────────────────────────────
# 9) Public API: clean + tokenize + spell‐correct
# ─────────────────────────────────────────────────────────────────────────────
def clean_and_tokenize_spellcheck(text: str) -> list[str]:
    """
    1) Run `clean_and_tokenize(text)` to get a list of “raw” tokens.
    2) For each raw token, call `_correct_token(...)` (which uses the cache +
       SpellChecker + custom_tokens).
    3) Return the final list of corrected tokens.
    """
    raw_tokens = clean_and_tokenize(text)
    return [_correct_token(tok) for tok in raw_tokens]


# ─────────────────────────────────────────────────────────────────────────────
# (Optionally, you could also expose a “just spellcheck a list of tokens”)
# ─────────────────────────────────────────────────────────────────────────────
def spellcheck_token_list(tokens: list[str]) -> list[str]:
    """
    If you already have a pre‐tokenized list of strings, you can call this
    function to run the same cache + SpellChecker logic on each token.
    """
    return [_correct_token(tok) for tok in tokens]
