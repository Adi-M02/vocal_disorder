# unit_testing/test_expand_unigrams.py
import json
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from gensim.models import Word2Vec

# ─────────────────────────────────────────────────────────────
# Import the module under test (adjust the path if needed)
# ─────────────────────────────────────────────────────────────
sys.path.append('../vocal_disorder')
import word2vec_expansion.expand_unigrams as script

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def small_corpus():
    # Keep it tiny and deterministic; we’ll use HS to avoid negative sampling noise.
    texts = [
        "alpha beta gamma",
        "alpha beta",
        "beta gamma",
        "alpha gamma",
        "delta epsilon",
        "epsilon delta",
    ]
    return [t.split() for t in texts]

def _train_and_save_w2v(tmp_path: Path, sg: int, filename: str, corpus):
    model_dir = tmp_path / "word2vec_08_07_16_41"
    model_dir.mkdir()
    model_path = model_dir / filename
    model = Word2Vec(
        sentences=corpus,
        vector_size=20,
        window=2,
        min_count=1,
        workers=1,
        sg=sg,           # 0 = CBOW, 1 = Skip-gram
        hs=1,            # hierarchical softmax (deterministic)
        negative=0,
        epochs=50,
    )
    model.save(str(model_path))
    return model_path, model_dir

@pytest.fixture()
def cbow_model_path(tmp_path, small_corpus):
    return _train_and_save_w2v(tmp_path, sg=0, filename="word2vec_cbow.model", corpus=small_corpus)

@pytest.fixture()
def skipgram_model_path(tmp_path, small_corpus):
    # filename lacks "skipgram" on purpose to exercise fallback via model.sg
    return _train_and_save_w2v(tmp_path, sg=1, filename="w2v.model", corpus=small_corpus)

@pytest.fixture(autouse=True)
def patch_process_text(monkeypatch):
    # Make token normalisation trivial & predictable for tests.
    monkeypatch.setattr(script, "process_text", lambda s: s.lower().split())
    yield

# Fake datetime to stabilize folder names
class _FixedDatetime(script.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2025, 8, 8, 12, 34, 56)

@pytest.fixture()
def freeze_time(monkeypatch):
    monkeypatch.setattr(script, "datetime", _FixedDatetime)
    yield

# ─────────────────────────────────────────────────────────────
# Unit tests: helpers
# ─────────────────────────────────────────────────────────────
def test_get_arch_from_filename(cbow_model_path):
    model_path, _ = cbow_model_path
    model = Word2Vec.load(str(model_path))
    arch = script.get_arch(model_path, model)
    assert arch == "cbow"

def test_get_arch_fallback_from_model(skipgram_model_path):
    model_path, _ = skipgram_model_path
    model = Word2Vec.load(str(model_path))
    # Filename has no "_skipgram.model", so fallback must use model.sg
    arch = script.get_arch(model_path, model)
    assert arch == "skipgram"

def test_most_similar_k_present_and_oov(cbow_model_path):
    model_path, _ = cbow_model_path
    model = Word2Vec.load(str(model_path))

    # Present
    res = script.most_similar_k(model, "alpha", k=3)
    assert isinstance(res, list)
    assert len(res) <= 3
    # results should be from vocab and not include the query itself
    for w in res:
        assert w in model.wv.key_to_index
        assert w != "alpha"

    # OOV
    assert script.most_similar_k(model, "not_in_vocab", k=5) == []

def test_build_output_path_format(freeze_time, tmp_path):
    out = script.build_output_path(tmp_path, arch="cbow", k=7)
    # path should be: <tmp>/<mm_dd_hh_mm>_expansion{arch}/topk_{k}.json
    assert out.parent.name == "08_08_12_34_expansioncbow"
    assert out.name == "topk_7.json"
    assert out.parent.exists()

# ─────────────────────────────────────────────────────────────
# Integration tests: main()
# ─────────────────────────────────────────────────────────────
class Args(types.SimpleNamespace):
    pass

@pytest.mark.parametrize("seeds_obj", [
    # list form
    ["alpha", "beta", "zeta_oov"],
    # dict form
    {"cat1": ["alpha"], "cat2": ["beta", "zeta_oov"]},
])
def test_main_writes_json_and_in_right_folder(
    freeze_time, cbow_model_path, tmp_path, seeds_obj
):
    model_path, model_dir = cbow_model_path

    # write seeds JSON
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(seeds_obj))

    # run
    args = Args(model=str(model_path), seed_json=str(seeds_path), topk=5)
    script.main(args)

    # expected folder & file
    exp_dir = model_dir / "08_08_12_34_expansioncbow"
    out_file = exp_dir / "topk_5.json"
    assert exp_dir.exists(), "Expected timestamp+arch folder not created"
    assert out_file.exists(), "Expected JSON output file not found"

    # validate contents
    data = json.loads(out_file.read_text())
    # Normalized keys (script lowercases via patched process_text)
    expected_keys = {"alpha", "beta", "zeta_oov"} if isinstance(seeds_obj, list) else {"alpha", "beta", "zeta_oov"}
    assert set(data.keys()) == expected_keys

    # alpha/beta present terms should have up to 5 neighbors; oov is empty list
    assert isinstance(data["alpha"], list) and all(w != "alpha" for w in data["alpha"])
    assert isinstance(data["beta"], list) and all(w != "beta" for w in data["beta"])
    assert data["zeta_oov"] == []

def test_topk_larger_than_vocab_ok(freeze_time, cbow_model_path, tmp_path):
    model_path, model_dir = cbow_model_path
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(["alpha"]))

    # K bigger than vocab-1 should not error; should truncate
    args = Args(model=str(model_path), seed_json=str(seeds_path), topk=999)
    script.main(args)

    out_file = (model_dir / "08_08_12_34_expansioncbow" / "topk_999.json")
    data = json.loads(out_file.read_text())
    assert "alpha" in data
    # neighbors cannot exceed vocab-1
    assert len(data["alpha"]) <= len(Word2Vec.load(str(model_path)).wv) - 1

def test_duplicate_seeds_collapse_in_output(freeze_time, cbow_model_path, tmp_path):
    model_path, model_dir = cbow_model_path
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(["alpha", "alpha", "ALPHA"]))

    args = Args(model=str(model_path), seed_json=str(seeds_path), topk=3)
    script.main(args)

    out_file = (model_dir / "08_08_12_34_expansioncbow" / "topk_3.json")
    data = json.loads(out_file.read_text())
    # dict keys deduplicate; with our lowercasing process_text, should be a single "alpha"
    assert list(data.keys()) == ["alpha"]

def test_arch_fallback_folder_name_for_skipgram(freeze_time, skipgram_model_path, tmp_path):
    model_path, model_dir = skipgram_model_path
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(["alpha"]))

    args = Args(model=str(model_path), seed_json=str(seeds_path), topk=2)
    script.main(args)

    # Folder should use "expansionskipgram" (inferred from model.sg)
    out_file = model_dir / "08_08_12_34_expansionskipgram" / "topk_2.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "alpha" in data
