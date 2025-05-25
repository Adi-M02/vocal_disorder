import re
import unicodedata




def clean_and_tokenize(text: str) -> list[str]:
    tokeniser   = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'/\-]*[A-Za-z0-9])?", re.I)
    url_pattern = re.compile(r'(?:https?://|www\.|[A-Za-z0-9\-]+\.(?:com|org|io|be)/)\S+', re.I)
    # remove everything except letters, numbers, apostrophe, hyphen, slash, or space
    junk_pattern = re.compile(r"[^A-Za-z0-9'/\- ]+")
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