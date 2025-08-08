"""
pytest suite for word2vec_expansion/split_manual_terms.py

What we cover
-------------
✔ max-ngram filtering (keeps ≤ N tokens)
✔ duplicate-removal logic
✔ seed-percent logic, incl. rounding & edge-cases
✔ output-file name & JSON structure
✔ invalid seed-percent outside [0, 1] raises SystemExit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# make the package importable when the tests run from anywhere
sys.path.append("../vocal_disorder")
from word2vec_expansion import split_manual_terms as splitter  # noqa: E402


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------
def run_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_terms: str,
    *,
    seed_percent: float,
    max_ngram: int,
):
    """
    Invoke splitter.main() inside an isolated temp directory.

    Parameters
    ----------
    tmp_path
        pytest fixture – per-test temp directory.
    monkeypatch
        pytest fixture – lets us stub functions / attributes.
    raw_terms
        Comma-separated phrase list to write into *terms.txt*.
    seed_percent, max_ngram
        CLI arguments to pass through.

    Returns
    -------
    seeds : list[str]
        Contents of the `"seed_terms"` key written by splitter.
    out_json : Path
        Actual JSON path produced (so we can check its name / content).
    """
    # 1) input file
    manual = tmp_path / "terms.txt"
    manual.write_text(raw_terms, encoding="utf-8")

    # 2) output dir
    outdir = tmp_path / "out"

    # 3) deterministic pipeline – replace process_text with simple str.split
    monkeypatch.setattr(splitter, "process_text", lambda term: term.split())

    # 4) deterministic shuffle – turn random.shuffle into a NO-OP
    monkeypatch.setattr(splitter.random, "shuffle", lambda x: None)

    # 5) fake CLI argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_manual_terms.py",
            "-i",
            str(manual),
            "-o",
            str(outdir),
            "-s",
            str(seed_percent),
            "-m",
            str(max_ngram),
        ],
    )

    # 6) run splitter & read output
    splitter.main()
    out_json = outdir / f"{max_ngram}_gram_seed_terms.json"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return data["seed_terms"], out_json


# ---------------------------------------------------------------------------
# max_ngram = 1  (order-agnostic check)
# ---------------------------------------------------------------------------
def test_max_ngram_1_any_order(tmp_path, monkeypatch):
    """
    max_ngram=1 keeps only 1-token phrases.
    Order is irrelevant because dedup now uses a set.
    """
    raw = "one,two three,four five six,seven eight,nine"
    seeds, _ = run_split(
        tmp_path,
        monkeypatch,
        raw_terms=raw,
        seed_percent=1.0,  # keep everything that survives
        max_ngram=1,
    )
    assert set(seeds) == {"one", "nine"}
    assert len(seeds) == 2                        # exactly two unique items
    assert len(seeds) == len(set(seeds))          # no duplicates


# ---------------------------------------------------------------------------
# max_ngram = 2, seed_percent = 0.5  (order-agnostic check)
# ---------------------------------------------------------------------------
def test_max_ngram_2_seed_half_any_order(tmp_path, monkeypatch):
    """
    After 2-gram filtering the unique pool is:
        {"one", "two three", "seven eight", "nine"}
    seed_percent = 0.5  → n_seed = 2.  We only check that
    we got ANY two distinct items from that pool.
    """
    raw = "one,two three,four five six,seven eight,nine"
    seeds, _ = run_split(
        tmp_path,
        monkeypatch,
        raw_terms=raw,
        seed_percent=0.5,
        max_ngram=2,
    )
    valid_pool = {"one", "two three", "seven eight", "nine"}
    assert set(seeds).issubset(valid_pool)
    assert len(seeds) == 2
    assert len(seeds) == len(set(seeds))


# ---------------------------------------------------------------------------
# Duplicate-removal
# ---------------------------------------------------------------------------
def test_duplicate_terms_removed(tmp_path, monkeypatch):
    """
    Duplicate phrases (after normalisation) must appear only once.
    """
    raw = "one,one,two,two,two two,two two"
    seeds, _ = run_split(
        tmp_path,
        monkeypatch,
        raw_terms=raw,
        seed_percent=1.0,
        max_ngram=2,
    )
    assert set(seeds) == {"one", "two", "two two"}
    assert len(seeds) == 3


# ---------------------------------------------------------------------------
# Edge-case: seed_percent == 0
# ---------------------------------------------------------------------------
def test_seed_percent_zero(tmp_path, monkeypatch):
    seeds, _ = run_split(
        tmp_path,
        monkeypatch,
        raw_terms="a,b,c",
        seed_percent=0.0,
        max_ngram=1,
    )
    assert seeds == []


# ---------------------------------------------------------------------------
# Output file naming & JSON structure
# ---------------------------------------------------------------------------
def test_output_filename_and_json_key(tmp_path, monkeypatch):
    seeds, out_json = run_split(
        tmp_path,
        monkeypatch,
        raw_terms="x,y",
        seed_percent=1.0,
        max_ngram=3,
    )
    assert out_json.name == "3_gram_seed_terms.json"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "seed_terms" in data
    assert data["seed_terms"] == seeds


# ---------------------------------------------------------------------------
# Invalid seed_percent → SystemExit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", ["-0.01", "1.01"])
def test_invalid_seed_percent_raises(tmp_path, monkeypatch, bad_value):
    manual = tmp_path / "terms.txt"
    manual.write_text("a,b", encoding="utf-8")
    outdir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_manual_terms.py",
            "-i",
            str(manual),
            "-o",
            str(outdir),
            "-s",
            bad_value,
            "-m",
            "1",
        ],
    )

    with pytest.raises(SystemExit):
        splitter.main()
