# tfidf_utils.py
"""
Train TF-IDF once, save artifacts, and expose a simple IDF lookup.

Artifacts saved in model_dir:
- vectorizer.pkl  (full scikit-learn TfidfVectorizer; optional but handy)
- idf_map.json    (term -> idf float)
- df_map.json     (term -> document frequency int)
- meta.json       (training metadata)

Usage:
    from tfidf_utils import train_and_store_tfidf, get_idf

    # Train once
    train_and_store_tfidf("models/tfidf_noburp")

    # Anywhere else
    val = get_idf("esophagitis", "models/tfidf_noburp")
    print(val)  # float or None if not found
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict
from functools import lru_cache

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Bring in your project helpers
sys.path.append("../vocal_disorder")
from query_mongo import return_documents
from utils.text_pipeline import process_text


def _normalize_tokens_to_text(tokens: list[str]) -> str:
    """Join pretokenized/lemmatized tokens into a space-separated string."""
    return " ".join(tokens)


def _normalize_term(term: str) -> str:
    """Normalize a single raw term using your pipeline to match training."""
    toks = process_text(term)  # -> list[str]
    return _normalize_tokens_to_text(toks)


def train_and_store_tfidf(
    model_dir: str | Path,
    db_name: str = "reddit",
    collection_name: str = "noburp_all",
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int | float = 5,
    max_df: float = 0.8,
    smooth_idf: bool = True,
    sublinear_tf: bool = True,
) -> Path:
    """
    Train TF-IDF on texts from Mongo (db_name, collection_name) and store artifacts.

    Parameters
    ----------
    model_dir : where to write artifacts
    db_name, collection_name : source for return_documents()
    ngram_range : e.g., (1,3) to get unigrams→trigrams
    min_df, max_df, smooth_idf, sublinear_tf : standard TfidfVectorizer params

    Returns
    -------
    Path to the model_dir with artifacts written.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & normalize documents ----
    raw_docs: Iterable[str] = return_documents(db_name, collection_name)
    # Materialize; TfidfVectorizer needs an indexable sequence
    cleaned_docs: list[list[str]] = [process_text(text) for text in raw_docs]
    texts: list[str] = [_normalize_tokens_to_text(toks) for toks in cleaned_docs]

    # ---- Fit TF-IDF ----
    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,                 # already normalized via process_text
        token_pattern=r"(?u)\b\w+\b",    # treat tokens as words (keeps numbers etc.)
        sublinear_tf=sublinear_tf,
        smooth_idf=smooth_idf,
        dtype=np.float32,
    )
    X = vect.fit_transform(texts)

    terms = vect.get_feature_names_out()
    idf_vals = vect.idf_.astype(float)

    # Document frequency (# of docs containing term)
    df_vals = (X.astype(bool).sum(axis=0).A1).astype(int)

    idf_map = {t: float(v) for t, v in zip(terms, idf_vals)}
    df_map = {t: int(v) for t, v in zip(terms, df_vals)}

    # ---- Save artifacts ----
    joblib.dump(vect, model_dir / "vectorizer.pkl")
    (model_dir / "idf_map.json").write_text(
        json.dumps(idf_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (model_dir / "df_map.json").write_text(
        json.dumps(df_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_docs": len(texts),
                "ngram_range": list(ngram_range),
                "min_df": min_df,
                "max_df": max_df,
                "smooth_idf": smooth_idf,
                "sublinear_tf": sublinear_tf,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return model_dir


@lru_cache(maxsize=32)
def _load_idf_map(idf_json_path: str | Path) -> Dict[str, float]:
    """Cached loader for the saved IDF map."""
    p = Path(idf_json_path)
    return json.loads(p.read_text(encoding="utf-8"))


def get_idf(
    term: str,
    model_dir: str | Path,
    normalize: bool = True,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Return the IDF value for a single term (unigram/bigram/trigram, etc.).

    Parameters
    ----------
    term : raw string, e.g., "esophagitis" or "upper esophageal sphincter"
    model_dir : directory that contains idf_map.json (output of training)
    normalize : if True, normalizes term via process_text to match training
    default : value to return if term not found (default: None)

    Returns
    -------
    float IDF value if present, else `default`.
    """
    idf_path = Path(model_dir) / "idf_map.json"
    idf_map = _load_idf_map(idf_path)

    key = _normalize_term(term) if normalize else term
    return idf_map.get(key, default)


# Optional: simple CLI for convenience
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train TF-IDF or query IDF.")
    ap.add_argument("--model_dir", "-o", required=True, help="Output / model dir")
    ap.add_argument("--train", action="store_true", help="Train and store TF-IDF")
    ap.add_argument("--db", default="reddit")
    ap.add_argument("--collection", default="noburp_all")
    ap.add_argument("--term", help="If provided without --train, returns its IDF")
    ap.add_argument("--normalize", action="store_true", help="Normalize input term")
    ap.add_argument("--min_df", type=float, default=5)
    ap.add_argument("--max_df", type=float, default=0.8)
    ap.add_argument("--ngrams", type=str, default="1,3",
                    help="ngram range as 'lo,hi' (e.g., '1,3')")
    args = ap.parse_args()

    if args.train:
        lo, hi = (int(x) for x in args.ngrams.split(","))
        out = train_and_store_tfidf(
            args.model_dir,
            db_name=args.db,
            collection_name=args.collection,
            ngram_range=(lo, hi),
            min_df=args.min_df,
            max_df=args.max_df,
        )
        print(f"Saved TF-IDF artifacts to: {out.resolve()}")
    elif args.term:
        val = get_idf(args.term, args.model_dir, normalize=args.normalize)
        print(val if val is not None else "NOT FOUND")
    else:
        ap.print_help()
