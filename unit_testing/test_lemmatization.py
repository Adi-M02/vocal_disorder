import sys
import pytest
import json

# adjust path so Python can find your modules
sys.path.append('../vocal_disorder')

# import the function under test
from utils.load_lemmatizer import load_lookup


def test_load_lookup_reads_json(tmp_path):
    # create a fake lookup JSON
    mapping = {
        "breathes": "breathe",
        "bloated": "bloat",
        "coughing": "cough"
    }
    file_path = tmp_path / "lemma_lookup.json"
    file_path.write_text(json.dumps(mapping), encoding='utf-8')

    # load via your function
    loaded = load_lookup(str(file_path))
    assert isinstance(loaded, dict)
    # it should exactly match the original mapping
    assert loaded == mapping


@pytest.mark.parametrize("token,expected", [
    # generic lemmatization
    ("breathes", "breathe"),
    ("bloated", "bloat"),
    ("coughing", "cough"),
    ("painful", "pain"),
    ("vomited", "vomit"),
    # gastrointestinal-specific
    ("nauseous", "nauseous"),      # maybe not in mapping -> fallback
    ("belched", "belch"),
    ("gurgling", "gurgle"),
    ("hiccuped", "hiccup"),
    # RCPD-specific
    ("burping", "burp"),
    ("gasping", "gasp"),
])
def test_lemma_lookup_mapping(tmp_path, token, expected):
    # construct a lookup map covering our test cases
    lookup_map = {
        "breathes": "breathe",
        "bloated": "bloat",
        "coughing": "cough",
        "painful": "pain",
        "vomited": "vomit",
        "belched": "belch",
        "gurgling": "gurgle",
        "hiccuped": "hiccup",
        "burping": "burp",
        "gasping": "gasp",
    }
    file_path = tmp_path / "lemma_lookup.json"
    file_path.write_text(json.dumps(lookup_map), encoding='utf-8')

    # load and apply mapping
    lm = load_lookup(str(file_path))
    # lookup_map.get(token, token) mimics your list comprehension
    result = lm.get(token, token)
    assert result == expected
