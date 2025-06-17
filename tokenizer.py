import re
import unicodedata

def clean_and_tokenize(text: str) -> list[str]:
    tokeniser   = re.compile(
        r"[A-Za-z\u00C0-\u024F0-9](?:[A-Za-z\u00C0-\u024F0-9'/\-]*[A-Za-z\u00C0-\u024F0-9])?",
        re.I
    )
    url_pattern = re.compile(r'(?:https?://|www\.|[A-Za-z0-9\-]+\.(?:com|org|io|be)/)\S+', re.I)
    # now allow apostrophes through, to strip them in the next step
    junk_pattern = re.compile(r"[^A-Za-z0-9'/\- ]+")

    # 1) normalize unicode, fix dashes & quotes
    clean = unicodedata.normalize("NFKC", text)
    clean = clean.replace("–", "-").replace("—", "-")
    clean = clean.replace("‘", "'").replace("’", "'")
    clean = clean.replace("…", "...")
    clean = clean.replace("\u200B", "").replace("\u00A0", " ")

    # 2) strip URLs
    clean = url_pattern.sub(" ", clean)

    # 3) preserve r/ and u/ 
    clean = re.sub(r"\b([ru])/", r"\1<SLASH>", clean, flags=re.I)

    # 4) collapse other slashes & hyphens
    clean = clean.replace("/", " ")
    clean = clean.replace("<SLASH>", "/")
    clean = clean.replace("-", " ")

    # 5) drop all junk except letters, numbers, apostrophes, hyphens, spaces
    clean = junk_pattern.sub(" ", clean)

    # 6) **strip apostrophes entirely** so "body's" → "bodys"
    clean = clean.replace("'", "")

    # 7) final tokenization on lowercase
    return tokeniser.findall(clean.lower())