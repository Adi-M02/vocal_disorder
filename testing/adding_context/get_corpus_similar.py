from typing import Optional, List
import pandas as pd
import re
import sys

sys.path.append("../vocal_disorder")  # adjust as needed
from utils.load_process_ngram_docs import process_ngram_docs
from utils.load_and_process_docs import process_all_noburp
from utils.load_json import load_json
# your existing funcs must already exist somewhere:
# - process_all_noburp(stoplist=False, lemmatize=False)
# - load_phrasers_from_dir(ngram_phraser_dir)
# - apply_ngrams(doc, (bigram, trigram))
# - remove_unigram_stopwords(doc)

def build_ngram_df(ngram_phraser_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - doc_id          : int
      - base_tokens     : list[str]   (tokenized 'base' form; no lemmatization)
      - base_text       : str         (space-joined base tokens)
      - ngram_tokens    : list[str]   (processed tokens with n-grams)
      - ngram_text      : str         (space-joined n-gram tokens)
      - ngram_token_set : set[str]    (for fast membership on n-gram tokens)
      - base_len        : int         (# of base tokens, useful for "longest")
    """
    # Base/tokenized docs (your note: "only tokenized form" comes from this call)
    base_docs = process_all_noburp(stoplist=False, lemmatize=False)

    # N-grammed docs (your existing pipeline)
    ngram_docs = process_ngram_docs(ngram_phraser_dir)

    if len(base_docs) != len(ngram_docs):
        raise ValueError(f"Mismatched lengths: base={len(base_docs)} vs ngram={len(ngram_docs)}")

    df = pd.DataFrame({
        "doc_id": range(len(base_docs)),
        "base_tokens": base_docs,
        "ngram_tokens": ngram_docs,
    })
    # Join tokens back into sentences/strings
    df["base_text"] = df["base_tokens"].apply(" ".join)
    df["ngram_text"] = df["ngram_tokens"].apply(" ".join)

    # Fast membership + lengths for sorting
    df["ngram_token_set"] = df["ngram_tokens"].apply(set)
    df["base_len"] = df["base_tokens"].apply(len)
    return df

def count_docs_containing(df: pd.DataFrame, term: str) -> int:
    """
    Number of documents whose N-GRAM token set contains the exact term.
    (Pass terms like 'able_to_burp' for n-gram matches.)
    """
    return int(df["ngram_token_set"].apply(lambda s: term in s).sum())

def sample_docs_containing(
    df: pd.DataFrame,
    term: str,
    k: int,
    *,
    random_state: Optional[int] = None,   # kept for compat; unused
    regex_text_fallback: bool = False,
) -> List[str]:
    """
    Return up to k *base-text* documents (List[str]) that contain `term`
    (matched primarily against n-gram tokens), sorted by base length desc.

    If `regex_text_fallback` is True and no n-gram token matches are found,
    tries regex on ngram_text first, then on base_text (with underscores→spaces).
    """
    # Primary: exact n-gram token membership
    mask = df["ngram_token_set"].apply(lambda s: term in s)
    hits = df[mask]

    if regex_text_fallback and hits.empty:
        # Try on the n-gram text as-is
        pat_ng = r"\b" + re.escape(term.strip()) + r"\b"
        hits = df[df["ngram_text"].str.contains(pat_ng, case=False, regex=True)]

        # If still empty, try replacing underscores with spaces and search base_text
        if hits.empty:
            term_spaced = term.replace("_", " ").strip()
            if term_spaced:
                pat_base = r"\b" + re.escape(term_spaced) + r"\b"
                hits = df[df["base_text"].str.contains(pat_base, case=False, regex=True)]

    if hits.empty:
        return []

    # Sort by base document length (descending) and return the base_text strings
    hits_sorted = hits.sort_values("base_len", ascending=False)
    return hits_sorted.head(k)["base_text"].tolist()

if __name__ == "__main__":
    df = build_ngram_df("testing/ngram_evals_test_no_digits/4")

    # Load expansions and derive the seed vocabulary
    all_expansions = load_json('testing/ngram_evals_test_no_digits/4/topk_25_min_cos_0.4_cbow.json')
    global_seed_vocab: List[str] = (
        sorted([str(s) for s, v in all_expansions.items() if isinstance(v, list) and v])
    )

    total = len(global_seed_vocab)

    failures = []
    for seed in global_seed_vocab:
        cnt = count_docs_containing(df, seed)
        if cnt < 3:
            failures.append((seed, cnt))
            print(f"[MISS] seed='{seed}'  count={cnt}  (< 3)")
        else:
            print(f"[OK]   seed='{seed}'  count={cnt}")

    print("\nSummary:")
    print(f"  Seeds checked: {total}")
    print(f"  Seeds meeting ≥3: {total - len(failures)}")
    print(f"  Seeds failing: {len(failures)}")

    if failures:
        print("\nFailing seeds (seed, count):")
        for s, c in failures:
            print(f"  {s}\t{c}")

    from collections import Counter

    # Build a doc-frequency map for ALL n-gram tokens in the corpus
    # (Much faster than df.apply(...) inside a loop)
    ngram_df_counts = Counter()
    for s in df["ngram_token_set"]:
        ngram_df_counts.update(s)

    # Collect all unique expansion terms (values) across seeds
    global_expansion_vocab: List[str] = sorted({
        str(term)
        for v in all_expansions.values()
        if isinstance(v, list)
        for term in v
        if isinstance(term, str) and term.strip()
    })

    # (Optional) exclude seeds themselves if you only want pure "new" terms
    # seed_set = set(map(str, all_expansions.keys()))
    # global_expansion_vocab = [t for t in global_expansion_vocab if t not in seed_set]

    # Verify coverage at threshold k=3 (match your seed check)
    exp_failures = []
    for term in global_expansion_vocab:
        cnt = ngram_df_counts.get(term, 0)
        if cnt < 3:
            exp_failures.append((term, cnt))
            print(f"[MISS] expansion='{term}'  count={cnt}  (< 3)")
        else:
            print(f"[OK]   expansion='{term}'  count={cnt}")

    print("\nExpansion summary:")
    print(f"  Unique expansions checked: {len(global_expansion_vocab)}")
    print(f"  Expansions meeting ≥3: {len(global_expansion_vocab) - len(exp_failures)}")
    print(f"  Expansions failing: {len(exp_failures)}")

    if exp_failures:
        print("\nFailing expansions (term, count):")
        for t, c in exp_failures:
            print(f"  {t}\t{c}")