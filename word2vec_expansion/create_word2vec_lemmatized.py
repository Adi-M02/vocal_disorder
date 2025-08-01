"""
usage python word2vec_expansion/create_word2vec_spellcheck.py

Train Word2Vec on Reddit posts with optional spell-checking and lemmatization lookup.

Adds lemmatization via a precomputed lookup table before training.
Uses the spellchecker to correct tokens 
"""
import sys
import os
import datetime
import time
import json
import argparse
from pathlib import Path

from gensim.models import Word2Vec

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
from spellchecker_folder.spellchecker import spellcheck_token_list
from utils.load_lemmatizer import load_lookup


def make_outdir(base_outdir: str, now: datetime.datetime) -> Path:
    """Create and return a timestamped output directory."""
    root = Path(base_outdir) if base_outdir else Path("word2vec_expansion")
    folder = root / now.strftime("word2vec_%m_%d_%H_%M")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_info(out_dir: Path, args: argparse.Namespace, now: datetime.datetime) -> None:
    """Write a JSON file with run arguments and timestamp."""
    info = {**vars(args), "timestamp": now.isoformat()}
    with open(out_dir / "info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)


def train_one_model(
    cleaned_docs: list[list[str]],
    out_dir: Path,
    args: argparse.Namespace,
    sg_flag: int,
    w2v_cls=Word2Vec,
    epochs: int = 5,
) -> Word2Vec:
    """Train and save a single Word2Vec model (CBOW or Skip-gram)."""
    model = w2v_cls(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=sg_flag,
        workers=max(1, os.cpu_count() - 1)
    )
    model.build_vocab(cleaned_docs)
    model.train(cleaned_docs, total_examples=len(cleaned_docs), epochs=epochs)

    name = "word2vec_cbow.model" if sg_flag == 0 else "word2vec_skipgram.model"
    path = out_dir / name
    model.save(str(path))
    return model


def run_training_pipeline(
    cleaned_docs: list[list[str]],
    args: argparse.Namespace,
    now: datetime.datetime | None = None,
    w2v_cls=Word2Vec,
) -> tuple[Word2Vec, Word2Vec, Path]:
    """
    Orchestrate directory creation, info writing, and training of CBOW and Skip-gram models.
    Returns the two trained models and the output directory.
    """
    now = now or datetime.datetime.now()
    out_dir = make_outdir(args.outdir, now)
    write_info(out_dir, args, now)

    cbow = train_one_model(cleaned_docs, out_dir, args, sg_flag=0, w2v_cls=w2v_cls)
    skipgram = train_one_model(cleaned_docs, out_dir, args, sg_flag=1, w2v_cls=w2v_cls)
    return cbow, skipgram, out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on r/noburp Reddit posts with lemmatization lookup."
    )
    parser.add_argument(
        "--lookup", type=str,
        default="testing/lemma_lookup.json",
        help="Path to JSON lemma lookup table."
    )
    parser.add_argument(
        "--vector_size", type=int, default=300,
        help="Embedding dimensionality."
    )
    parser.add_argument(
        "--window", type=int, default=7,
        help="Context window size."
    )
    parser.add_argument(
        "--min_count", type=int, default=3,
        help="Ignore tokens with total frequency lower than this."
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory to save models and info."
    )
    args = parser.parse_args()

    # Load lemma lookup map
    lookup_map = load_lookup(args.lookup)
    logging = print  # simple print for progress

    # 2) Fetch raw Reddit docs
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    logging(f"Number of documents fetched: {len(docs)}")

    # 3) Tokenize → lemmatize → spellcheck → lemmatize
    cleaned_docs: list[list[str]] = []
    for text in docs:
        toks = clean_and_tokenize(text)
        toks = [lookup_map.get(tok, tok) for tok in toks]
        toks = spellcheck_token_list(toks)
        toks = [lookup_map.get(tok, tok) for tok in toks]
        cleaned_docs.append(toks)
    logging(f"Tokenized & lemmatized {len(cleaned_docs)} documents")

    # 5‑8) Run the training pipeline
    cbow, skipgram, out_dir = run_training_pipeline(cleaned_docs, args)
    logging(f"Training complete. Models saved in {out_dir}")


if __name__ == "__main__":
    main()
