import sys
import os
import pytest
import pickle

# ensure we can import the spellchecker module and load actual custom_tokens
sys.path.append('../vocal_disorder')
import spellchecker_folder.spellchecker as sc
from spellchecker_folder.spellchecker import spellcheck_token_list, _correct_token

# Dummy SPELL checker to control suggestions
class DummySpell:
    def __init__(self, result_map=None):
        # result_map: mapping from token to suggestion or None
        self.result_map = result_map or {}
        self.called = []
    def correction(self, token):
        # record each token asked and return configured suggestion
        self.called.append(token)
        return self.result_map.get(token)

# Fixture: clear only the in-memory cache before each test, but keep custom_tokens loaded
@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    # reset disk cache
    monkeypatch.setattr(sc, '_disk_cache', {})
    yield

# 1) Verify custom_tokens loaded from file
def test_custom_terms_loaded_from_file():
    assert isinstance(sc.custom_tokens, set)
    # There should be at least some custom RCPD-related tokens
    assert 'rcpd' in sc.custom_tokens or len(sc.custom_tokens) > 0

# 2) Custom tokens bypass correction
def test_custom_token_returns_unchanged(monkeypatch):
    # pick an actual custom token from file
    token = next(iter(sc.custom_tokens))
    dummy = DummySpell({token: 'wrong'})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = _correct_token(token)
    assert result == token
    # SPELL.correction should not be called
    assert dummy.called == []

# 3) Cached tokens returned without calling SPELL
def test_cached_token(monkeypatch):
    sc._disk_cache['hello'] = 'hi'
    dummy = DummySpell({'hello': 'ignored'})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = spellcheck_token_list(['hello'])
    assert result == ['hi']
    assert dummy.called == []

# 4) Corrupted cache entry should be removed and re-corrected
def test_corrupted_cache(monkeypatch):
    # ensure no custom-token collision for 'notacustom'
    monkeypatch.setattr(sc, 'custom_tokens', set())
    class CorruptCache(dict):
        def __getitem__(self, key):
            raise pickle.UnpicklingError()
    monkeypatch.setattr(sc, '_disk_cache', CorruptCache({'badly': 'x'}))
    dummy = DummySpell({'badly': 'fixed'})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = _correct_token('badly')
    assert result == 'fixed'
    assert sc._disk_cache.get('badly') == 'fixed'

# 5) Tokens containing digits are returned and cached unchanged
def test_digit_tokens_are_cached_and_unchanged(monkeypatch):
    dummy = DummySpell({'h2o': 'water'})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = spellcheck_token_list(['h2o', '2025'])
    assert result == ['h2o', '2025']
    assert sc._disk_cache['h2o'] == 'h2o'
    assert sc._disk_cache['2025'] == '2025'
    assert dummy.called == []

# 6) RCPD-specific misspelling corrected by SPELL
def test_spell_suggestion_used_rcpd(monkeypatch):
    monkeypatch.setattr(sc, 'custom_tokens', set())
    dummy = DummySpell({'berping': 'burping'})  # common miscoded symptom
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = spellcheck_token_list(['berping'])
    assert result == ['burping']
    assert sc._disk_cache['berping'] == 'burping'
    assert dummy.called == ['berping']

# 7) Mixed behaviors including cache, digit, RCPD misspelling, fallback
def test_mixed_behaviors_rcpd(monkeypatch):
    # Mixed behaviors: cache, digit skipping, RCPD-specific correction, fallback
    monkeypatch.setattr(sc, 'custom_tokens', set())  # no custom tokens

    # prime disk cache for 'old'
    sc._disk_cache['old'] = 'new'
    dummy = DummySpell({'bloaring': 'bloating', 'junk': None})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    tokens = ['keep', 'old', 'h2o', 'bloaring', 'junk']
    result = spellcheck_token_list(tokens)

    # 'keep' -> SPELL called, fallback to 'keep'
    # 'old' -> from cache
    # 'h2o' -> digit skip
    # 'bloaring' -> SPELL -> 'bloating'
    # 'junk' -> SPELL returns None -> 'junk'
    assert result == ['keep', 'new', 'h2o', 'bloating', 'junk']

    # ensure cache entries
    assert sc._disk_cache['keep'] == 'keep'
    assert sc._disk_cache['h2o'] == 'h2o'
    assert sc._disk_cache['bloaring'] == 'bloating'
    assert sc._disk_cache['junk'] == 'junk'

    # SPELL should be called for keep, bloaring, junk
    assert dummy.called == ['keep', 'bloaring', 'junk']

# 8) Parameterized common misspellings including RCPD terms
@pytest.mark.parametrize("token,expected", [
    ("definately", "definitely"),
    ("occured", "occurred"),
    ("seperate", "separate"),
    ("cricopharyngal", "cricopharyngeal"),
    ("dysphagiaa", "dysphagia"),
    ("gasx", "gasx"),  # unknown, fallback to itself
    ("badly", "badly"),
])
def test_common_misspellings_rcpd(monkeypatch, token, expected):
    monkeypatch.setattr(sc, 'custom_tokens', set())
    dummy = DummySpell({token: expected})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = _correct_token(token)
    assert result == expected

# 9) Correctly spelled RCPD terms unchanged
@pytest.mark.parametrize("token", [
    "rcpd", "dysphagia", "botox", "esophagus", "bloating", "burping"
])
def test_correct_spellings_unchanged(monkeypatch, token):
    # ensure no suggestions and not in custom_tokens
    dummy = DummySpell({})
    monkeypatch.setattr(sc, 'SPELL', dummy)
    monkeypatch.setattr(sc, 'custom_tokens', set())

    result = _correct_token(token)
    assert result == token
@pytest.mark.parametrize("token,expected", [
    ('bloatting','bloating'),
    ('burpingg','burping'),
    ('reguritation','regurgitation'),
    ('emisis','emesis'),
    ('nausee','nausea'),
    ('gurgiling','gurgling'),
    ('heartburnn','heartburn'),
    ('naurea','nausea'),
])
def test_extended_misspellings_rcpd(monkeypatch, token, expected):
    monkeypatch.setattr(sc, 'custom_tokens', set())
    dummy = DummySpell({token: expected})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = _correct_token(token)
    assert result == expected

# 10) Correctly spelled RCPD terms unchanged
@pytest.mark.parametrize("token", [
    'rcpd','dysphagia','botox','esophagus','bloating','burping'
])
def test_correct_spellings_unchanged(monkeypatch, token):
    monkeypatch.setattr(sc, 'custom_tokens', set())
    dummy = DummySpell({})
    monkeypatch.setattr(sc, 'SPELL', dummy)

    result = _correct_token(token)
    assert result == token