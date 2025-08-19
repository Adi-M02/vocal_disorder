#!/usr/bin/env python3
# word2phrase_eval.py
"""
Evaluate n-gram generation using ONLY Word2Phrase (gensim Phrases/Phraser).
- Loads lemmatized+spellchecked unigram tokens from process_all_noburp(stoplist=False)
- Trains bigram (and optional trigram) phrasers
- Samples documents and writes input vs output token lists (JSONL)
- Optionally saves trained phrasers (timestamped dir with manifest) and a transformed corpus (JSONL)

Examples:
  python word2phrase_eval.py --min-count 5 --threshold 10 --passes 2 --connector light \
    --k-samples 20 --out-jsonl eval.jsonl --save-phrasers --save-transformed transformed.jsonl

Notes:
- Remove unigram stopwords AFTER transformation.
- Parameters to tune: min_count, threshold, passes(1|2), connector list, scoring.
"""

from __future__ import annotations
import sys, json, argparse, random
from pathlib import Path
from typing import List, Iterable, Optional, Dict, Any, Tuple, Set, Union
from collections import Counter
from datetime import datetime

# Your corpus loader
sys.path.append('../vocal_disorder')
from utils.load_and_process_docs import process_all_noburp  # returns List[List[str]]

# gensim word2phrase
from gensim.models.phrases import Phrases, Phraser

# ---------- connectors ----------
LIGHT_CONNECTORS: Set[str] = {
    "of","to","in","for","with","on","at","from","and","or","as","per"
}
HEAVY_CONNECTORS: Set[str] = LIGHT_CONNECTORS | {
    "by","than","via","into","over","under","about","through","without"
}

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
    scoring: str = "default",  # "default" ~ word2phrase, "npmi" also available
    delimiter: str = "_",
    connectors: Optional[Set[str]] = None,
    passes: int = 2
) -> Tuple[Phraser, Optional[Phraser]]:
    """
    Train bigram (and optional trigram) Phrasers.
    Returns (bigram_phraser, trigram_phraser_or_None)
    """
    # Bigram
    phrases2 = Phrases(
        docs,
        min_count=min_count,
        threshold=threshold,
        delimiter=delimiter,                 # must match token type (str)
        scoring=scoring,
        connector_words=frozenset(connectors) if connectors else None,
    )
    bigram = Phraser(phrases2)

    if passes <= 1:
        return bigram, None

    # Trigram: train on bigrammed docs
    bigrammed: Iterable[List[str]] = (bigram[doc] for doc in docs)
    phrases3 = Phrases(
        bigrammed,
        min_count=min_count,                 # keep equal; change if you prefer lower for trigrams
        threshold=threshold,                 # reuse; tune separately if desired
        delimiter=delimiter,
        scoring=scoring,
        connector_words=frozenset(connectors) if connectors else None,
    )
    trigram = Phraser(phrases3)
    return bigram, trigram

# ---------- transform ----------
def transform_doc(tokens: List[str], bigram: Phraser, trigram: Optional[Phraser]) -> List[str]:
    out = bigram[tokens]
    if trigram is not None:
        out = trigram[out]
    return out

# ---------- sampling & summary ----------
def sample_eval(
    docs: List[List[str]],
    bigram: Phraser,
    trigram: Optional[Phraser],
    k_samples: int,
    seed: int,
    out_jsonl: Optional[str] = None
) -> Dict[str, Any]:
    rnd = random.Random(seed)
    idxs = rnd.sample(range(len(docs)), min(k_samples, len(docs)))

    # Choose output path
    if not out_jsonl:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_jsonl = str(Path(__file__).parent / f"word2phrase_test_outputs/word2phrase_eval_{ts}.jsonl")

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")  # fresh file

    total_merged = 0
    phrase_counter = Counter()

    with out_path.open("a", encoding="utf-8") as f:
        for j, i in enumerate(idxs):
            inp = docs[i]
            out = transform_doc(inp, bigram, trigram)
            created = [t for t in out if "_" in t]
            total_merged += len(created)
            phrase_counter.update(created)
            row = {
                "sample_id": j,
                "doc_index": i,
                "input": inp,
                "output": out,
                "phrases": created,
                "num_merged": len(created),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "out_jsonl": str(out_path),
        "k_samples": len(idxs),
        "total_phrases_in_samples": total_merged,
        "unique_phrases_in_samples": len(phrase_counter),
        "top_phrases_in_samples": phrase_counter.most_common(30),
    }
    return summary

# ---------- saving (timestamped dir + manifest) ----------
def save_run_artifacts(
    *,
    bigram: Phraser,
    trigram: Optional[Phraser],
    args: argparse.Namespace,
    connectors: Optional[Set[str]],
    n_docs: int,
    sample_jsonl_path: Optional[str],
    base_dir: str = "word2phrase_saved_phrasers",
    delimiter: str = "_"
) -> Path:
    """
    Creates a timestamped dir under base_dir, saves bigram/trigram phrasers, and writes manifest.json.
    Returns the run directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bigram_path = run_dir / "bigram.phraser"
    trigram_path = run_dir / "trigram.phraser"
    manifest_path = run_dir / "manifest.json"

    bigram.save(str(bigram_path))
    if trigram is not None:
        trigram.save(str(trigram_path))

    manifest: Dict[str, Any] = {
        "created_at": ts,
        "generator": "word2phrase_eval.py",
        "params": {
            "min_count": args.min_count,
            "threshold": args.threshold,
            "scoring": args.scoring,
            "passes": args.passes,
            "delimiter": delimiter,
            "connector_preset": args.connector,
            "connector_file": args.connector_file or "",
            "k_samples": args.k_samples,
            "seed": args.seed,
            "limit_docs": args.limit_docs,
        },
        "connectors": sorted(list(connectors)) if connectors else None,
        "phrasers": {
            "bigram": str(bigram_path.name),
            "trigram": str(trigram_path.name) if trigram is not None else None,
        },
        "docs_info": {
            "n_docs": n_docs
        },
        "artifacts": {
            "sample_eval_jsonl": sample_jsonl_path or None
        }
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir

# ---------- transform corpus saving ----------
def save_transformed(docs: List[List[str]], bigram: Phraser, trigram: Optional[Phraser], out_path: str):
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for toks in docs:
            merged = transform_doc(toks, bigram, trigram)
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")

# ---------- loader to re-use a saved run ----------
def load_phrasers_from_dir(run_dir: Union[str, Path]) -> Tuple[Phraser, Optional[Phraser]]:
    """
    Load (bigram, trigram?) phrasers from a saved run directory.
    Prefers manifest.json to locate files; falls back to default filenames.
    """
    run_dir = Path(run_dir)
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        bigram_name = m.get("phrasers", {}).get("bigram", "bigram.phraser")
        trigram_name = m.get("phrasers", {}).get("trigram", "trigram.phraser")
        bigram_path = run_dir / bigram_name
        trigram_path = run_dir / trigram_name if trigram_name else None
    else:
        bigram_path = run_dir / "bigram.phraser"
        trigram_path = run_dir / "trigram.phraser"

    if not bigram_path.exists():
        raise FileNotFoundError(f"Missing bigram phraser at {bigram_path}")

    bigram = Phraser.load(str(bigram_path))
    trigram = Phraser.load(str(trigram_path)) if trigram_path and Path(trigram_path).exists() else None
    return bigram, trigram

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Word2Phrase n-gram generation on the noburp corpus")
    p.add_argument("--min-count", type=int, default=5, help="Minimum occurrence for a phrase candidate")
    p.add_argument("--threshold", type=float, default=5.0, help="Higher -> fewer phrases (Mikolov threshold)")
    p.add_argument("--scoring", type=str, default="default", choices=["default", "npmi"],
                   help="Word2Phrase original score ('default') or 'npmi'")
    p.add_argument("--passes", type=int, default=2, choices=[1,2], help="1=only bigrams, 2=also trigrams")
    p.add_argument("--connector", type=str, default="light", choices=["none","light","heavy"],
                   help="Preset connector list (discourages merges through these words)")
    p.add_argument("--connector-file", type=str, default="", help="Optional path to custom connector list (one word per line)")
    p.add_argument("--k-samples", type=int, default=100000, help="Number of sample docs to output")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit-docs", type=int, default=0, help="Use only first N docs (speed-up)")
    p.add_argument("--out-jsonl", type=str, default="", help="Where to write sample input/output JSONL. "
                   "If omitted, a timestamped file is created next to this script.")
    p.add_argument("--save-phrasers", action="store_true", help="If set, save phrasers + manifest to timestamped dir")
    p.add_argument("--save-dir", type=str, default="word2phrase_saved_phrasers",
                   help="Base directory for saved runs (used only if --save-phrasers is set)")
    p.add_argument("--save-transformed", type=str, default="", help="If set, save transformed corpus tokens as JSONL")
    return p.parse_args()

def main():
    args = parse_args()

    print("[info] loading docs with stoplist=False ...")
    docs = process_all_noburp(stoplist=False)
    if args.limit_docs and args.limit_docs > 0:
        docs = docs[:args.limit_docs]
        print(f"[info] limiting to {len(docs)} docs")

    connectors = resolve_connectors(args.connector, args.connector_file)
    print(f"[info] training Word2Phrase: min_count={args.min_count}, threshold={args.threshold}, "
          f"scoring={args.scoring}, passes={args.passes}, connectors="
          f"{'None' if not connectors else f'{len(connectors)} words'}")

    bigram, trigram = train_phrasers(
        docs,
        min_count=args.min_count,
        threshold=args.threshold,
        scoring=args.scoring,
        delimiter="_",                # str delimiter to match str tokens
        connectors=connectors,
        passes=args.passes,
    )

    summary = sample_eval(
        docs, bigram, trigram,
        k_samples=args.k_samples,
        seed=args.seed,
        out_jsonl=args.out_jsonl if args.out_jsonl else None
    )
    print("[info] sample summary:", json.dumps(summary, ensure_ascii=False))

    # Save run artifacts (phrasers + manifest) if requested
    if args.save_phrasers:
        run_dir = save_run_artifacts(
            bigram=bigram,
            trigram=trigram,
            args=args,
            connectors=connectors,
            n_docs=len(docs),
            sample_jsonl_path=summary.get("out_jsonl"),
            base_dir=args.save_dir,
            delimiter="_"
        )
        print(f"[info] saved run to: {run_dir}")

    # Optionally save transformed corpus
    if args.save_transformed:
        save_transformed(docs, bigram, trigram, args.save_transformed)
        print(f"[info] saved transformed corpus to: {args.save_transformed}")

# -------- public API: apply_ngrams that accepts a path OR phrasers --------
def apply_ngrams(
    tokens: List[str],
    model: Union[str, Path, Phraser, Tuple[Phraser, Optional[Phraser]]]
) -> List[str]:
    """
    Apply a Word2Phrase model to a token list with default behavior.
    - If `model` is str/Path: path to a saved run directory (with manifest + phrasers).
    - If `model` is a single bigram Phraser: applies bigrams.
    - If `model` is (bigram, trigram): applies bigrams, then trigrams.
    """
    # Directory path case
    if isinstance(model, (str, Path)):
        bigram, trigram = load_phrasers_from_dir(model)
        out = list(bigram[tokens])
        if trigram is not None:
            out = list(trigram[out])
        return out

    # Phraser(s) case
    if isinstance(model, Phraser):
        return list(model[tokens])
    bigram, trigram = model
    out = list(bigram[tokens])
    if trigram is not None:
        out = list(trigram[out])
    return out

if __name__ == "__main__":
    main()
