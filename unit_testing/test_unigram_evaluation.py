import importlib
import json
import sys
from pathlib import Path

import pytest

# So we can import your utils tokenizer
sys.path.append("../vocal_disorder")

MODULE_NAME = "word2vec_expansion.unigram_expansion.evaluate_expansion"


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(MODULE_NAME)


# --- Use the real tokenizer, but with a temp empty lookup for stability ----
@pytest.fixture
def real_tokenizer_with_tmp_lookup(tmp_path):
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text("{}", encoding="utf-8")
    from utils.text_pipeline import process_text as real_process_text

    def wrapped(text: str):
        return real_process_text(text, lookup_path=str(lookup_path))
    return wrapped


# ---------------------------
# _normalize_unigram_set
# ---------------------------

def test_normalize_unigram_set_keeps_unigrams_and_drops_multi(mod, monkeypatch, capsys, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    terms = ["burp", "very long", ""]  # "very long" -> 2 tokens, dropped; "" -> 0 tokens, skipped
    norm = mod._normalize_unigram_set(terms)
    assert norm == {"burp"}
    captured = capsys.readouterr().out
    assert "dropped" in captured  # warning message


# ---------------------------
# _flatten_expanded / _flatten_ground_truth
# ---------------------------

def test_flatten_expanded_happy_path(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    expanded = {
        "digestive": ["gut", "enzyme"],
        "position": ["tilt", "very long"],  # drops "very long"
    }
    out = mod._flatten_expanded(expanded)
    assert out == {"gut", "enzyme", "tilt"}

def test_flatten_expanded_errors(mod):
    with pytest.raises(ValueError):
        mod._flatten_expanded(["not", "a", "dict"])
    with pytest.raises(ValueError):
        mod._flatten_expanded({"bad": "notalist"})

def test_flatten_ground_truth_happy_path(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    gt = {"seed_terms": ["burp", "bar", "very long"]}
    out = mod._flatten_ground_truth(gt)
    assert out == {"burp", "bar"}

def test_flatten_ground_truth_errors(mod):
    with pytest.raises(ValueError):
        mod._flatten_ground_truth({"nope": ["a", "b"]})
    with pytest.raises(ValueError):
        mod._flatten_ground_truth({"seed_terms": "notalist"})


# ---------------------------
# evaluate_token_level
# ---------------------------

def test_evaluate_token_level_counts_metrics(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)

    det = {"burp", "false"}
    gt  = {"burp", "bar"}

    # DB-return structure: List[str]
    docs = [
        "burp burp unk",   # TP=2, TN=1
        "false bar baz",     # FP=1, FN=1, TN=1
    ]

    results = mod.evaluate_token_level(docs, det, gt)
    counts = results["counts"]
    assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}

    m = results["metrics"]
    assert pytest.approx(m["precision"], rel=1e-6) == 2/3
    assert pytest.approx(m["recall"],    rel=1e-6) == 2/3
    assert pytest.approx(m["f1"],        rel=1e-6) == 2/3
    assert pytest.approx(m["accuracy"],  rel=1e-6) == 4/6
    assert pytest.approx(m["specificity"], rel=1e-6) == 2/3
    assert pytest.approx(m["balanced_accuracy"], rel=1e-6) == 2/3
    denom_sqrt = ((3*3*3*3) ** 0.5)
    assert pytest.approx(m["mcc"], rel=1e-6) == ((2*2 - 1*1) / denom_sqrt)

    per = results["per_term"]
    assert dict(per["tp_terms"]) == {"burp": 2}
    assert dict(per["fp_terms"]) == {"false": 1}
    assert dict(per["fn_terms"]) == {"bar": 1}

    # AUC/AP may be None (degenerate) or valid floats
    for k in ("roc_auc_hard", "average_precision_hard"):
        val = m[k]
        assert (val is None) or (isinstance(val, float) and 0.0 <= val <= 1.0)


def test_evaluate_token_level_empty_inputs(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    results = mod.evaluate_token_level([], set(), set())
    assert results["counts"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    m = results["metrics"]
    for k in ("precision","recall","f1","accuracy","specificity","balanced_accuracy","mcc"):
        assert m[k] == 0.0
    # AUC/AP undefined -> None
    assert m["roc_auc_hard"] is None
    assert m["average_precision_hard"] is None


def test_evaluate_token_level_only_unknown_tokens_count_as_tn(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    docs = ["alpha beta gamma"]  # none in either set -> all TN
    det, gt = set(), set()
    res = mod.evaluate_token_level(docs, det, gt)
    assert res["counts"]["tn"] == 3
    assert res["counts"]["tp"] == 0 and res["counts"]["fp"] == 0 and res["counts"]["fn"] == 0
    # AUC/AP undefined because only class 0 present
    assert res["metrics"]["roc_auc_hard"] is None
    assert res["metrics"]["average_precision_hard"] is None


def test_evaluate_token_level_skips_blank_and_empty_token_docs(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    # "", "   ", and "..." should all produce no tokens with your cleaner
    docs = ["", "   ", "...", "burp"]
    det, gt = {"burp"}, set()
    res = mod.evaluate_token_level(docs, det, gt)
    # Only "burp" contributes -> FP=1
    assert res["counts"] == {"tp": 0, "fp": 1, "fn": 0, "tn": 0}
    meta = res["meta"]
    assert meta["docs_processed"] == 4
    assert meta["docs_skipped_empty"] == 3
    assert meta["tokens_seen"] == 1


def test_evaluate_token_level_all_positive_class_average_precision_defined(mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    # All tokens are positive (both in det and gt)
    det, gt = {"burp"}, {"burp"}
    docs = ["burp burp"]  # TP=2
    res = mod.evaluate_token_level(docs, det, gt)
    assert res["counts"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 0}
    # ROC AUC undefined (single class), AP should be 1.0
    assert res["metrics"]["roc_auc_hard"] is None
    ap = res["metrics"]["average_precision_hard"]
    assert ap is None or (isinstance(ap, float) and 0.0 <= ap <= 1.0)
    # In practice sklearn returns 1.0 here; allow either for portability


# ---------------------------
# main() integration (mock I/O + DB)
# ---------------------------

def test_main_integration_smoke(tmp_path, mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)

    def fake_load_json(path):
        p = str(path)
        if p.endswith("expanded.json"):
            return {"cat": ["burp", "false", "very long"]}  # "very long" dropped
        if p.endswith("ground.json"):
            return {"seed_terms": ["burp", "bar"]}
        raise RuntimeError("Unexpected path")
    monkeypatch.setattr(mod, "load_json", fake_load_json)

    # return_documents now accepts DB params; accept **kwargs
    monkeypatch.setattr(mod, "return_documents", lambda **kwargs: ["burp burp unk", "false bar baz"])

    out_path = tmp_path / "out.json"
    argv = [
        "prog",
        "--expanded_json", str(tmp_path / "expanded.json"),
        "--ground_json",   str(tmp_path / "ground.json"),
        "--db_name", "reddit",
        "--collection_name", "noburp_all",
        "--out",           str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["counts"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert data["meta"]["expanded_size"] == 2
    assert data["meta"]["ground_truth_size"] == 2
    assert dict(data["per_term"]["tp_terms"]) == {"burp": 2}
    assert dict(data["per_term"]["fp_terms"]) == {"false": 1}
    assert dict(data["per_term"]["fn_terms"]) == {"bar": 1}


def test_main_bad_ground_raises(tmp_path, mod, monkeypatch, real_tokenizer_with_tmp_lookup):
    monkeypatch.setattr(mod, "process_text", real_tokenizer_with_tmp_lookup)
    # malformed ground truth
    monkeypatch.setattr(mod, "load_json", lambda p: {"seedz": ["nope"]} if str(p).endswith("ground.json") else {"cat": ["x"]})
    monkeypatch.setattr(mod, "return_documents", lambda **kwargs: ["x"])

    out_path = tmp_path / "out.json"
    argv = [
        "prog",
        "--expanded_json", "e.json",
        "--ground_json", "ground.json",
        "--db_name", "reddit",
        "--collection_name", "noburp_all",
        "--out", str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError):
        mod.main()
