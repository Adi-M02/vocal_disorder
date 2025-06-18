"""
evaluate the performance of word2vec expansion terms on manual text
usage:
  python word2vec_expansion/evaluate_expansions_lemmatized.py \
    --expansion_path <path> \
    --manual_dir <dir> \
    [--spellcheck] \
    [--lookup <lemma_lookup.json>] \
    [--ngram <N|<=N>]
"""
import sys
import os
import re
import json
import logging
import argparse
from collections import Counter
from typing import Optional, Tuple, List, Dict

# allow importing your project's tokenizer
sys.path.append("../vocal_disorder")
from tokenizer import clean_and_tokenize

from spellchecker_folder.spellchecker import spellcheck_token_list

from query_mongo import return_documents
import datetime



def parse_ngram_filter(arg: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse the --ngram argument into a (min_ngram, max_ngram) pair.
      * "3"    → (3, 3)
      * "<=2"  → (1, 2)
      * "<= 3" → (1, 3)
    """
    if arg is None:
        return None

    s = arg.strip()
    # <=N  (allow a space after <=)
    m = re.match(r'^<=\s*(\d+)$', s)
    if m:
        n = int(m.group(1))
        return (1, n)

    # exactly N
    m = re.match(r'^(\d+)$', s)
    if m:
        n = int(m.group(1))
        return (n, n)

    raise argparse.ArgumentTypeError(
        f"Invalid --ngram value: '{arg}'.  Use 'N' or '<=N'."
    )


def load_expansion_terms(
    path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn
) -> List[str]:
    """
    Load and tokenize expansion terms, filtering by ngram_filter if provided.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    terms: List[str] = []
    for value in data.values():
        candidates = value if isinstance(value, list) else [value]
        for term in candidates:
            tokens = tok_fn(term)
            if not tokens:
                continue
            L = len(tokens)
            if ngram_filter is None or (ngram_filter[0] <= L <= ngram_filter[1]):
                terms.append(" ".join(tokens))
    return terms


def load_manual_terms(
    path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn
) -> List[str]:
    """
    Load comma-separated manual terms, tokenize and filter by ngram_filter.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().lower()
    candidates = [t.strip() for t in raw.split(",") if t.strip()]

    cleaned: List[str] = []
    for term in candidates:
        tokens = tok_fn(term)
        if not tokens:
            continue
        L = len(tokens)
        if ngram_filter is None or (ngram_filter[0] <= L <= ngram_filter[1]):
            cleaned.append(" ".join(tokens))
    return cleaned


def load_user_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().lower()
    return [u.strip() for u in raw.split(",") if u.strip()]


def evaluate_terms_performance(
    docs: List[str],
    manual_terms_path: str,
    expansion_terms_path: str,
    ngram_filter: Optional[Tuple[int,int]],
    tok_fn,
    lemmatize: bool = False,
    lemma_map: Optional[Dict[str,str]] = None
) -> dict:
    """
    Evaluate precision/recall/F1/accuracy of expansion terms vs. manual terms.
    """
    # wrap tokenizer if lemmatization requested
    if lemmatize and lemma_map:
        base_tok = tok_fn
        def tok(text: str) -> List[str]:
            return [ lemma_map.get(t, t) for t in base_tok(text) ]
    else:
        tok = tok_fn

    def norm(term: str) -> str:
        return " ".join(tok(term))

    manual_terms    = load_manual_terms(manual_terms_path, ngram_filter, tok)
    expansion_terms = load_expansion_terms(expansion_terms_path, ngram_filter, tok)
    
    universe       = set(manual_terms) | set(expansion_terms)

    TP = FP = TN = FN = 0
    fp_counter = Counter()
    fn_counter = Counter()
    tp_counter = Counter()

    for doc in docs:
        tokens = tok(doc)
        positions = {}
        for term in universe:
            term_toks = term.split()
            L = len(term_toks)
            for i in range(len(tokens)-L+1):
                if tokens[i:i+L] == term_toks:
                    positions.setdefault(term, []).append((i, i+L))

        used = [False]*len(tokens)
        matched = set()
        for term in sorted(positions, key=lambda t: -len(t.split())):
            for (start, end) in positions[term]:
                if not any(used[start:end]):
                    matched.add(term)
                    for j in range(start, end):
                        used[j] = True
                    break

        # --- new coverage logic ---
        def covers(matched_terms: set[str], target: str) -> bool:
            t_toks = target.split()
            L = len(t_toks)
            for m in matched_terms:
                m_toks = m.split()
                if len(m_toks) == L and m == target:
                    return True
                if len(m_toks) > L:
                    for k in range(len(m_toks) - L + 1):
                        if m_toks[k:k+L] == t_toks:
                            return True
            return False

        manual_found    = {t for t in manual_terms    if covers(matched, t)}
        expansion_found = {t for t in expansion_terms if covers(matched, t)}

        for term in universe:
            m = term in manual_found
            e = term in expansion_found
            if m and e:
                TP += 1
                tp_counter[term] += 1
            elif not m and e:
                FP += 1
                fp_counter[term] += 1
            elif m and not e:
                FN += 1
                fn_counter[term] += 1
            else:
                TN += 1

    total     = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    accuracy  = (TP + TN) / total if total else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    base, _ = os.path.splitext(expansion_terms_path)
    ts = datetime.datetime.now().strftime("%m_%d_%H_%M")
    out_path = f"{base}_evaluation_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Expansion JSON: {expansion_terms_path}\n")
        if ngram_filter is None:
            f.write("No n-gram filtering applied\n")
        else:
            f.write(f"Filtered terms to n-grams in [{ngram_filter[0]}, {ngram_filter[1]}]\n")
        f.write(f"=== Evaluation Across {len(docs)} Documents ===\n")
        f.write(f"Total terms evaluated: {total}\n")
        f.write(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall:    {recall:.3f}\n")
        f.write(f"F1 Score:  {f1:.3f}\n")
        f.write(f"Accuracy:  {accuracy:.3f}\n\n")
        if tp_counter:
            total_tp = sum(tp_counter.values())
            f.write(f"True Positives ({total_tp} occurrences across {len(tp_counter)} terms):\n")
            for term, cnt in tp_counter.most_common():
                f.write(f"{term}: {cnt}\n")
            f.write("\n")
        if fp_counter:
            total_fp = sum(fp_counter.values())
            f.write(f"False Positives ({total_fp} occurrences across {len(fp_counter)} terms):\n")
            for term, cnt in fp_counter.most_common():
                f.write(f"{term}: {cnt}\n")
            f.write("\n")

        if fn_counter:
            total_fn = sum(fn_counter.values())
            f.write(f"False Negatives ({total_fn} occurrences across {len(fn_counter)} terms):\n")
            for term, cnt in fn_counter.most_common():
                f.write(f"{term}: {cnt}\n")
            f.write("\n")

    print(f"\nResults -- Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
    print(f"Details written to {out_path}")
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expansion JSON against manual terms, with optional n-gram filtering and lemmatization."
    )
    parser.add_argument("--expansion_path", required=True,
        help="Path to the expansion JSON file.")
    parser.add_argument("--manual_dir", required=True,
        help="Directory with manual_terms.txt and users.txt")
    parser.add_argument("--spellcheck", action="store_true",
        help="Use the spell-checking tokenizer.")
    parser.add_argument("--lookup", type=str, default=None,
        help="Path to lemma-lookup JSON; if set, lemmatize tokens via this mapping.")
    parser.add_argument("--ngram", type=str, default=None,
        help="Filter by n-gram length. Use 'N' for exactly N or '<=N' for up to N.")

    args = parser.parse_args()

    if args.spellcheck:
        print("Using spell-checking tokenizer")
        def tok_fn(text):
            return spellcheck_token_list(clean_and_tokenize(text))
    else:
        tok_fn = clean_and_tokenize
    print(f"→ Using {'spell-checking' if args.spellcheck else 'vanilla'} tokenizer")

    # optionally wrap with lemma lookup
    if args.lookup:
        with open(args.lookup, 'r', encoding='utf-8') as f:
            lemma_map = json.load(f)
        base_fn = tok_fn
        def tok_lemmatize(text: str):
            return [lemma_map.get(tok, tok) for tok in base_fn(text)]
        tok_fn = tok_lemmatize
        print(f"→ Applying lemma lookup from {args.lookup}")

    ngram_filter = parse_ngram_filter(args.ngram)

    # verify manual_dir
    manual_terms_path = os.path.join(args.manual_dir, "manual_terms.txt")
    users_path        = os.path.join(args.manual_dir, "users.txt")
    for p in (manual_terms_path, users_path):
        if not os.path.isfile(p):
            parser.error(f"{p} not found.")

    users = load_user_list(users_path)
    if not users:
        parser.error("No users listed in users.txt")

    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        filter_users=users
    )
    logging.info("Fetched %d documents for %d users", len(docs), len(users))

    evaluate_terms_performance(
        docs,
        manual_terms_path,
        args.expansion_path,
        ngram_filter,
        tok_fn
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()