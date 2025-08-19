"""
Coverage-only evaluation (train on FULL corpus, evaluate on user slice):
- Train Word2Phrase on the entire corpus (process_all_noburp).
- Build produced vocabulary (apply phrasers to the entire corpus; includes unigrams & merged n-grams).
- Load an evaluation slice (posts from --users) and count occurrences of manual terms in that slice.
- Compute coverage (unique & weighted) overall and by n-gram length (1,2,3+), weighted by slice counts.
- Save outputs under: <out-folder>/<threshold>/
    - bigram.phraser
    - trigram.phraser (if passes=2)
    - metrics.json
    - missed_terms.json
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from typing import List, Iterable, Optional, Dict, Any, Tuple, Set, Union
from datetime import datetime
from collections import Counter, defaultdict

# ---------- project imports ----------
sys.path.append('../vocal_disorder')
from utils.load_and_process_docs import process_all_noburp  # returns List[List[str]]
from utils.text_pipeline import process_text               # tokenizer/lemmatizer
from query_mongo import return_documents                   # raw texts

# ---------- gensim ----------
from gensim.models.phrases import Phrases, Phraser

# ---------- connector presets ----------
LIGHT_CONNECTORS: Set[str] = {
    "of","to","in","for","with","on","at","from","and","or","as","per"
}
HEAVY_CONNECTORS: Set[str] = LIGHT_CONNECTORS | {
    "by","than","via","into","over","under","about","through","without"
}

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def resolve_connectors(arg: Optional[str], path: Optional[str]) -> Optional[Set[str]]:
    if path:
        txt = Path(path).read_text(encoding="utf-8").splitlines()
        return {w.strip() for w in txt if w.strip()}
    if not arg or arg == "none":
        return None
    if arg == "light":
        return set(LIGHT_CONNECTORS)
    if arg == "heavy":
        return set(HEAVY_CONNECTORS)
    raise ValueError(f"Unknown connector preset: {arg}")

# ---------- training ----------
def train_phrasers(
    docs: List[List[str]],
    min_count: int,
    threshold: float,
    scoring: str = "default",
    delimiter: str = "_",
    connectors: Optional[Set[str]] = None,
    passes: int = 2
) -> Tuple[Phraser, Optional[Phraser]]:
    phrases2 = Phrases(
        docs,
        min_count=min_count,
        threshold=threshold,
        delimiter=delimiter,
        scoring=scoring,
        connector_words=frozenset(connectors) if connectors else None,
    )
    bigram = Phraser(phrases2)

    if passes <= 1:
        return bigram, None

    bigrammed: Iterable[List[str]] = (bigram[doc] for doc in docs)
    phrases3 = Phrases(
        bigrammed,
        min_count=min_count,
        threshold=threshold,
        delimiter=delimiter,
        scoring=scoring,
        connector_words=frozenset(connectors) if connectors else None,
    )
    trigram = Phraser(phrases3)
    return bigram, trigram

def transform_doc(tokens: List[str], bigram: Phraser, trigram: Optional[Phraser]) -> List[str]:
    out = bigram[tokens]
    if trigram is not None:
        out = trigram[out]
    return out

# ---------- IO helpers ----------
def _load_users(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--users file not found: {path}")
    users: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # handle "u/username"
        if s.lower().startswith("u/"):
            s = s[2:]
        # also allow comma-separated on same line
        for part in s.split(","):
            part = part.strip()
            if part:
                users.append(part)
    print(f"[info] loaded {len(users)} users from {path} (sample: {users[:5]})")
    return users or None

def _parse_seed_terms(path: str) -> List[str]:
    """
    Load seed terms from a JSON file with structure:
      { "seed_terms": ["burp_door_open", "out_of_pocket", ...] }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--seed-terms file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "seed_terms" not in data or not isinstance(data["seed_terms"], list):
        raise ValueError("Seed terms JSON must be an object with a 'seed_terms' list.")
    # Coerce to strings and strip
    return [str(t).strip() for t in data["seed_terms"] if str(t).strip()]

def _normalize_term_to_seq(term: str) -> Optional[List[str]]:
    """
    Normalize a seed term to a token SEQUENCE.
    Supports both space-separated and underscore-joined inputs.
    Example: 'out_of_pocket' -> ['out','of','pocket']
    """
    # Treat underscores as word separators before pipeline normalization
    term = term.replace("_", " ")
    toks = process_text(term, stoplist=False)
    return toks if toks else None

def _normalize_terms_to_seqs(terms_raw: List[str]) -> Dict[str, List[str]]:
    """
    Return a dict: joined_form ('able_to_burp') -> token sequence ['able','to','burp'].
    Unique by joined_form (first occurrence wins).
    """
    joined_to_seq: Dict[str, List[str]] = {}
    for t in terms_raw:
        seq = _normalize_term_to_seq(t)
        if not seq:
            continue
        joined = seq[0] if len(seq) == 1 else "_".join(seq)
        if joined not in joined_to_seq:
            joined_to_seq[joined] = seq
    return joined_to_seq

def _build_phraser_vocabulary_corpuswide(
    train_docs: List[List[str]],
    bigram: Phraser,
    trigram: Optional[Phraser]
) -> Set[str]:
    """Apply phrasers to the FULL training corpus and return all observed tokens."""
    vocab: Set[str] = set()
    for toks in train_docs:
        merged = transform_doc(toks, bigram, trigram)
        vocab.update(merged)
    return vocab

# ---------- counting occurrences in the EVAL slice ----------
def _count_term_occurrences_in_docs(
    eval_docs: List[List[str]],
    term_seqs: Dict[str, List[str]],
) -> Counter:
    """
    Count how many times each manual term (normalized to sequence) occurs in the evaluation docs.
    Matches are contiguous token sequences (exact match). Overlaps allowed.
    Returns Counter keyed by joined_form ('able_to_burp') with integer counts.
    """
    # index patterns by first token for speed
    start_index: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for joined, seq in term_seqs.items():
        if not seq:
            continue
        start_index[seq[0]].append((joined, seq))

    counts = Counter()
    for toks in eval_docs:
        n = len(toks)
        if n == 0:
            continue
        i = 0
        while i < n:
            first = toks[i]
            if first in start_index:
                for joined, seq in start_index[first]:
                    L = len(seq)
                    if L <= n - i and toks[i:i+L] == seq:
                        counts[joined] += 1
            i += 1
    return counts

def _ngram_len_from_joined(tok: str) -> str:
    n = tok.count("_") + 1
    return "1" if n == 1 else "2" if n == 2 else "3+"

def _coverage_from_eval_counts(
    eval_counts: Counter,
    produced_vocab: Set[str],
) -> Dict[str, Any]:
    """
    Coverage using only terms that appear in the EVAL slice (eval_counts>0).
    """
    unique_present = {t for t, c in eval_counts.items() if c > 0}
    total_unique = len(unique_present)
    total_weighted = sum(eval_counts.values())

    covered_unique = {t for t in unique_present if t in produced_vocab}
    covered_weighted = sum(eval_counts[t] for t in covered_unique)

    buckets = ["1", "2", "3+"]
    uniq_tot_by = {b: 0 for b in buckets}
    uniq_cov_by = {b: 0 for b in buckets}
    w_tot_by    = {b: 0 for b in buckets}
    w_cov_by    = {b: 0 for b in buckets}

    for t in unique_present:
        b = _ngram_len_from_joined(t)
        uniq_tot_by[b] += 1
        w_tot_by[b]    += eval_counts[t]
        if t in produced_vocab:
            uniq_cov_by[b] += 1
            w_cov_by[b]    += eval_counts[t]

    def safe_rate(num, den): return (num / den) if den else 0.0

    missed = [(t, eval_counts[t], _ngram_len_from_joined(t))
              for t in unique_present if t not in produced_vocab]
    missed.sort(key=lambda x: (-x[1], x[0]))

    return {
        "overall": {
            "unique": {
                "covered": len(covered_unique),
                "total": total_unique,
                "rate": safe_rate(len(covered_unique), total_unique),
            },
            "weighted": {
                "covered_count": covered_weighted,
                "total_count": total_weighted,
                "rate": safe_rate(covered_weighted, total_weighted),
            },
        },
        "by_ngram": {
            b: {
                "unique": {
                    "covered": uniq_cov_by[b],
                    "total": uniq_tot_by[b],
                    "rate": safe_rate(uniq_cov_by[b], uniq_tot_by[b]),
                },
                "weighted": {
                    "covered_count": w_cov_by[b],
                    "total_count": w_tot_by[b],
                    "rate": safe_rate(w_cov_by[b], w_tot_by[b]),
                },
            } for b in buckets
        },
        "top_missed": [
            {"term": t, "count": c, "ngram": b}
            for (t, c, b) in missed[:200]
        ],
        # add full missed list here for convenience (not capped)
        "missed_all": [
            {"term": t, "count": eval_counts[t], "ngram": _ngram_len_from_joined(t)}
            for t in unique_present if t not in produced_vocab
        ],
    }

# ---------- arg parsing ----------
def parse_args():
    p = argparse.ArgumentParser(description="Coverage-only evaluation (train on full corpus, evaluate on user slice).")
    # training params (for FULL corpus)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--scoring", type=str, default="default", choices=["default", "npmi"])
    p.add_argument("--passes", type=int, default=2, choices=[1, 2])
    p.add_argument("--connector", type=str, default="light", choices=["none", "light", "heavy"])
    p.add_argument("--connector-file", type=str, default="")
    p.add_argument("--limit-train-docs", type=int, default=0, help="Limit full-corpus docs for training (0 = no limit)")
    # evaluation slice
    p.add_argument("--users", type=str, default="vocabulary_evaluation/manual_terms_7_12/users.txt", help="Optional path to newline- or comma-separated usernames")
    p.add_argument("--limit-eval-docs", type=int, default=0, help="Limit docs in evaluation slice (0 = no limit)")
    # evaluation input/output
    p.add_argument("--seed-terms", type=str, default="vocabulary_evaluation/manual_terms_7_12/all_terms/1_to_3_gram_seed_terms.json", help="Path to seed terms file")
    p.add_argument("--out-folder", type=str, required=True, help="Base folder to write outputs into")
    # keeping these for CLI compatibility; they are no-ops now
    p.add_argument("--metrics-json", type=str, default="", help=argparse.SUPPRESS)
    # saving phrasers (optional) – now always saved into out-folder
    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()

    # Prepare run dir: <out-folder>/<threshold>/
    # Use a clean, consistent folder name for the threshold
    thr_name = f"{args.threshold:g}"
    run_dir = Path(args.out_folder) / thr_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output directory: {run_dir.resolve()}")

    # A) Load FULL corpus for training (always)
    print("[info] loading FULL corpus with process_all_noburp(stoplist=False) ...")
    train_docs = process_all_noburp(stoplist=False)
    if args.limit_train_docs and args.limit_train_docs > 0:
        train_docs = train_docs[:args.limit_train_docs]
        print(f"[info] limiting training to {len(train_docs)} docs")
    print(f"[info] full-corpus docs loaded for training: {len(train_docs)}")

    if not train_docs or all(len(t) == 0 for t in train_docs):
        raise SystemExit("No training documents loaded (or all empty). Check your corpus loader.")

    # B) Train phrasers on FULL corpus
    connectors = resolve_connectors(args.connector, args.connector_file)
    print(f"[info] training Word2Phrase on FULL corpus: min_count={args.min_count}, threshold={args.threshold}, "
          f"scoring={args.scoring}, passes={args.passes}, connectors="
          f"{'None' if not connectors else f'{len(connectors)} words'}")
    bigram, trigram = train_phrasers(
        train_docs,
        min_count=args.min_count,
        threshold=args.threshold,
        scoring=args.scoring,
        delimiter="_",
        connectors=connectors,
        passes=args.passes,
    )

    # C) ALWAYS save phrasers into the run dir
    bigram.save(str(run_dir / "bigram.phraser"))
    if trigram is not None:
        trigram.save(str(run_dir / "trigram.phraser"))

    # D) Build produced vocabulary by applying to FULL corpus
    produced_vocab = _build_phraser_vocabulary_corpuswide(train_docs, bigram, trigram)
    print(f"[info] produced_vocab size (FULL corpus): {len(produced_vocab)}")

    # E) Load EVALUATION SLICE (users) and tokenize
    if args.users:
        print(f"[info] loading EVAL slice from Mongo for users file: {args.users}")
        users = _load_users(args.users)
        eval_raw = return_documents(
            db_name="reddit",
            collection_name="noburp_all",
            filter_users=users,
        )
        if args.limit_eval_docs and args.limit_eval_docs > 0:
            eval_raw = eval_raw[:args.limit_eval_docs]
        print(f"[info] EVAL slice raw texts: {len(eval_raw)}")
        eval_docs = [process_text(text, stoplist=False) for text in eval_raw if isinstance(text, str) and text]
        print(f"[info] EVAL slice tokenized docs: {len(eval_docs)}, non-empty: {sum(1 for t in eval_docs if t)}")
    else:
        print("[info] no --users provided; evaluating on FULL corpus")
        eval_docs = train_docs

    if not eval_docs or all(len(t) == 0 for t in eval_docs):
        raise SystemExit("No EVAL documents loaded (or all empty). Check your users file / query.")

    # F) Load + normalize seed terms to sequences and joined forms
    terms_raw = _parse_seed_terms(args.seed_terms)
    joined_to_seq = _normalize_terms_to_seqs(terms_raw)
    print(f"[info] seed terms: raw={len(terms_raw)}, normalized_unique={len(joined_to_seq)}")

    print(f"[info] seed terms: raw={len(terms_raw)}, normalized_unique={len(joined_to_seq)}")
    # G) Count occurrences of manual terms IN THE EVAL SLICE (sequence matching; includes unigrams)
    eval_counts = _count_term_occurrences_in_docs(eval_docs, joined_to_seq)
    total_occ = sum(eval_counts.values())
    present_terms = sum(1 for k, v in eval_counts.items() if v > 0)
    print(f"[info] eval occurrences counted: total={total_occ}, unique_present={present_terms}")

    # H) Coverage metrics: only terms present in the EVAL slice contribute to weights
    coverage = _coverage_from_eval_counts(eval_counts, produced_vocab)

    # I) Write outputs into run dir
    metrics_payload = {
        "created_at": _ts(),
        "args": {
            "min_count": args.min_count,
            "threshold": args.threshold,
            "scoring": args.scoring,
            "passes": args.passes,
            "connector_preset": args.connector,
            "connector_file": args.connector_file or "",
            "limit_train_docs": args.limit_train_docs,
            "users": args.users,
            "limit_eval_docs": args.limit_eval_docs,
            "seed_terms": str(Path(args.seed_terms).resolve()),   # <-- renamed
        },
        "counts": {
            "produced_vocab_full_corpus": len(produced_vocab),
            "eval_terms_unique_present": present_terms,
            "eval_terms_total_occurrences": total_occ,
            "seed_terms_normalized_unique": len(joined_to_seq),   # <-- renamed
        },
        "coverage": {
            "overall": coverage["overall"],
            "by_ngram": coverage["by_ngram"],
            "top_missed": coverage["top_missed"],
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # missed_terms.json (full list, not capped)
    missed_items = coverage.get("missed_all", [])
    missed_payload = {
        "created_at": _ts(),
        "threshold": args.threshold,
        "total_missed_unique": len(missed_items),
        "total_missed_weighted": int(sum(item["count"] for item in missed_items)),
        "items": missed_items,  # [{"term": ..., "count": ..., "ngram": "2"/"3+"}, ...]
    }
    (run_dir / "missed_terms.json").write_text(json.dumps(missed_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[info] wrote outputs to: {run_dir}")

if __name__ == "__main__":
    main()
