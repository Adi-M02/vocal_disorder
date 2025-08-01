import sys
import os
import json
import datetime
import pytest
from pathlib import Path

# Make sure the refactored module is importable
sys.path.append('../vocal_disorder')
from word2vec_expansion import create_word2vec_lemmatized as script

# --- Fixtures and helpers --- #
@pytest.fixture
def dummy_args(tmp_path):
    """Return an argparse.Namespace with expected args as instance attributes."""
    import argparse
    return argparse.Namespace(
        vector_size=50,
        window=5,
        min_count=1,
        outdir=str(tmp_path)
    )

@pytest.fixture
def cleaned_docs():
    # simple dummy token lists
    return [['alpha', 'beta'], ['gamma']]

class DummyW2V:
    def __init__(self, **kwargs):
        # record init kwargs
        self.kwargs = kwargs
        self.built = False
        self.trained = None
        self.saved_path = None

    def build_vocab(self, docs):
        # record that build_vocab was called
        self.built = docs

    def train(self, docs, total_examples, epochs):
        # record training call
        self.trained = {
            'docs': docs,
            'total_examples': total_examples,
            'epochs': epochs
        }

    def save(self, path):
        # simulate saving by writing an empty file
        self.saved_path = path
        Path(path).write_text('')

# --- Tests --- #

def test_make_outdir(tmp_path):
    # fixed datetime
    now = datetime.datetime(2025, 8, 1, 12, 30, 0)
    base = str(tmp_path / 'base')
    out_dir = script.make_outdir(base, now)

    expected = Path(base) / now.strftime('word2vec_%m_%d_%H_%M')
    assert out_dir == expected
    assert out_dir.exists() and out_dir.is_dir()


def test_write_info(tmp_path, dummy_args):
    now = datetime.datetime(2025, 8, 1, 12, 30, 0)
    out_dir = tmp_path / 'info_test'
    out_dir.mkdir()

    script.write_info(out_dir, dummy_args, now)
    info_file = out_dir / 'info.json'
    assert info_file.exists()

    info = json.loads(info_file.read_text())
    # ensure args fields are present
    assert info['vector_size'] == dummy_args.vector_size
    assert info['window'] == dummy_args.window
    assert 'timestamp' in info
    assert info['timestamp'].startswith(now.isoformat())


def test_train_one_model_cbow(tmp_path, cleaned_docs, dummy_args):
    # Test sg_flag=0 (CBOW)
    out_dir = tmp_path / 'train_test'
    out_dir.mkdir()

    model = script.train_one_model(
        cleaned_docs,
        out_dir,
        dummy_args,
        sg_flag=0,
        w2v_cls=DummyW2V,
        epochs=3
    )

    # Check constructor kwargs
    assert model.kwargs['vector_size'] == dummy_args.vector_size
    assert model.kwargs['window'] == dummy_args.window
    assert model.kwargs['min_count'] == dummy_args.min_count
    assert model.kwargs['sg'] == 0

    # Check build/train recorded
    assert model.built == cleaned_docs
    assert model.trained['total_examples'] == len(cleaned_docs)
    assert model.trained['epochs'] == 3

    # Check file was saved
    expected_path = out_dir / 'word2vec_cbow.model'
    assert expected_path.exists()
    assert model.saved_path == str(expected_path)


def test_train_one_model_skipgram(tmp_path, cleaned_docs, dummy_args):
    # Test sg_flag=1 (Skip-gram)
    out_dir = tmp_path / 'train_test_sg'
    out_dir.mkdir()

    model = script.train_one_model(
        cleaned_docs,
        out_dir,
        dummy_args,
        sg_flag=1,
        w2v_cls=DummyW2V
    )

    # Check skip-gram flag
    assert model.kwargs['sg'] == 1
    # Check file name
    expected_path = out_dir / 'word2vec_skipgram.model'
    assert expected_path.exists()
    assert model.saved_path == str(expected_path)


def test_run_training_pipeline(tmp_path, cleaned_docs, dummy_args):
    # Use a fixed now and DummyW2V to test the full pipeline
    now = datetime.datetime(2025, 8, 1, 12, 0, 0)

    cbow, skipgram, out_dir = script.run_training_pipeline(
        cleaned_docs,
        dummy_args,
        now=now,
        w2v_cls=DummyW2V
    )

    # out_dir should match dummy_args.outdir + timestamped folder
    expected = Path(dummy_args.outdir) / now.strftime('word2vec_%m_%d_%H_%M')
    assert out_dir == expected

    # Info file exists
    info_file = out_dir / 'info.json'
    assert info_file.exists()
    data = json.loads(info_file.read_text())
    assert data['vector_size'] == dummy_args.vector_size

    # Models are DummyW2V instances with correct sg
    assert isinstance(cbow, DummyW2V)
    assert isinstance(skipgram, DummyW2V)
    assert cbow.kwargs['sg'] == 0
    assert skipgram.kwargs['sg'] == 1

    # Saved files exist
    assert (out_dir / 'word2vec_cbow.model').exists()
    assert (out_dir / 'word2vec_skipgram.model').exists()
