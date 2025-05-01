# evaluate_vocab.py
import json, re, sys, os
from itertools import chain
from pathlib import Path
from typing import List, Dict, Tuple, Set

import pandas as pd

# ── project-specific imports ────────────────────────────────────────────
sys.path.append(os.path.abspath("vocabulary_evaluation"))
sys.path.append(os.path.abspath("."))          # so that analyze_users is found
from analyze_users import (
    prepare_user_dataframe_multi, preprocess_terms
)

# ── helpers ─────────────────────────────────────────────────────────────
def _normalize_term(term: str) -> str:
    term = re.sub(r'[^a-z0-9\s\-]', '', term.lower())
    return re.sub(r'\s+', ' ', term).strip()

def _normalize_text(txt: str) -> str:
    txt = re.sub(r'[^a-z0-9\s\-]', ' ', txt.lower())
    return re.sub(r'\s+', ' ', txt).strip()

def _match_terms(terms: Set[str], text: str) -> Set[str]:
    found = set()
    for t in terms:
        if re.search(rf'(?<!\w){re.escape(t)}(?!\w)', text):
            found.add(t)
    return found

def _load_ground_truth(path: Path) -> List[str]:
    terms = Path(path).read_text(encoding="utf-8").split(",")
    cleaned = [_normalize_term(t.strip()) for t in preprocess_terms(terms)]
    return [t for t in cleaned if t]

def _load_vocab_terms(path: Path) -> Set[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = chain.from_iterable(obj["vocabulary"].values())
    return {_normalize_term(t) for t in preprocess_terms(raw)}

# ── public API ──────────────────────────────────────────────────────────
def evaluate_vocab(
    vocab_json_path: str | Path,
    ground_truth_path: str | Path = "vocabulary_evaluation/manual_terms.txt",
    usernames: List[str] | None = None,
    write_back: bool = True,
) -> Dict[str, float]:
    """
    Run precision / recall / accuracy evaluation for a vocabulary JSON file.

    Returns a dict with the metrics.  If *write_back* is True, the metrics
    are saved into the JSON's ["metadata"]["evaluation"] field.
    """
    usernames = usernames or []
    vocab_json_path = Path(vocab_json_path)
    ground_truth_path = Path(ground_truth_path)

    # 1) load terms
    vocab_terms = _load_vocab_terms(vocab_json_path)
    ground_terms = _load_ground_truth(ground_truth_path)

    # 2) build text corpus for the supplied users
    df = prepare_user_dataframe_multi(usernames)
    corpus = _normalize_text(" ".join(df["content"].fillna("").tolist()))

    # 3) term matching
    candidate_terms = set(ground_terms) | vocab_terms
    terms_in_text = _match_terms(candidate_terms, corpus)

    gt_found, vb_found = (
        set(ground_terms) & terms_in_text,
        vocab_terms & terms_in_text,
    )
    tp = len(gt_found & vb_found)
    fn = len(gt_found - vb_found)
    fp = len(vb_found - gt_found)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    accuracy  = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    jaccard   = len(gt_found & vb_found) / len(gt_found | vb_found) if gt_found | vb_found else 0.0

    metrics = {
        "evaluated_on_users": usernames,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "jaccard_similarity": round(jaccard, 4),
    }

    # 4) optionally write back into the JSON
    if write_back:
        data = json.loads(vocab_json_path.read_text(encoding="utf-8"))
        data.setdefault("metadata", {}).update({"evaluation": metrics})
        vocab_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics

# ── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="Evaluate a vocabulary JSON against ground-truth terms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            Single file:
                python evaluate_vocab.py vocab_output/expanded_vocab.json

            With custom ground-truth file and users:
                python evaluate_vocab.py vocab.json -g custom.txt -u user1 user2
        """),
    )
    parser.add_argument("json", help="Path to the vocabulary JSON file")
    parser.add_argument("-g", "--ground", default="vocabulary_evaluation/manual_terms.txt",
                        help="Ground-truth term list")
    parser.add_argument("-u", "--usernames", nargs="*", default=[],
                        help="Usernames whose posts are used as the evaluation corpus")
    parser.add_argument("--no-write", action="store_true",
                        help="Do NOT write metrics back into the JSON")

    args = parser.parse_args()
    m = evaluate_vocab(
        args.json,
        ground_truth_path=args.ground,
        usernames=args.usernames,
        write_back=not args.no_write,
    )
    print(json.dumps(m, indent=2))
    