import sys
import json
from pathlib import Path
import pytest
sys.path.append('../vocal_disorder')
from word2vec_expansion import split_manual_terms as splitter

def run_split(tmp_path, monkeypatch, raw_terms, seed_percent, max_ngram):
    # 1) create input file
    manual = tmp_path / "terms.txt"
    manual.write_text(raw_terms, encoding="utf-8")

    # 2) define output dir
    outdir = tmp_path / "out"

    # 3) stub out process_text -> simple space-split
    monkeypatch.setattr(splitter, "process_text", lambda term: term.split())

    # 4) fix shuffle for determinism
    monkeypatch.setattr(splitter.random, "shuffle", lambda x: None)

    # 5) set sys.argv and run
    monkeypatch.setattr(sys, "argv", [
        "split_manual_terms.py",
        "-i", str(manual),
        "-o", str(outdir),
        "-s", str(seed_percent),
        "-m", str(max_ngram),
    ])
    splitter.main()

    # 6) read output JSON
    out_file = outdir / f"{max_ngram}_gram_seed_terms.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    return data["seed_terms"]


def test_max_ngram_1(tmp_path, monkeypatch):
    """
    With max_ngram=1, only single-token terms survive.
    """
    raw = "one,two three,four five six,seven eight,nine"
    seeds = run_split(tmp_path, monkeypatch, raw, seed_percent=1.0, max_ngram=1)
    # processed_terms -> ["one"],["two","three"],...; only ["one"],["nine"] remain
    assert seeds == ["one", "nine"]


def test_max_ngram_2_and_seed_percent(tmp_path, monkeypatch):
    """
    With max_ngram=2 and seed_percent=0.5, we should get the first half
    of the filtered list (no shuffle).
    """
    raw = "one,two three,four five six,seven eight,nine"
    seeds = run_split(tmp_path, monkeypatch, raw, seed_percent=0.5, max_ngram=2)
    # filtered -> ["one"],["two three"],["seven eight"],["nine"]
    # seed_percent=0.5 => n_seed=int(4*0.5)=2 => ["one","two three"]
    assert seeds == ["one", "two three"]


def test_invalid_seed_percent(tmp_path, monkeypatch):
    """
    seed_percent outside [0,1] should cause SystemExit.
    """
    manual = tmp_path / "terms.txt"
    manual.write_text("a,b", encoding="utf-8")
    outdir = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", [
        "split_manual_terms.py",
        "-i", str(manual),
        "-o", str(outdir),
        "-s", "-0.1",    # invalid
        "-m", "1",
    ])
    with pytest.raises(SystemExit):
        splitter.main()
