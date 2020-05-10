import pytest
from typing import List
from thinc.api import Model
from spacy.tokens import Doc
from spacy.vocab import Vocab

from spacy_transformers.tagger import transformer_linear_v1
from spacy_transformers.types import TransformerData
from .util import DummyTransformer


@pytest.fixture
def trf_width() -> int:
    return 768


@pytest.fixture
def trf_depth() -> int:
    return 4


@pytest.fixture
def output_width() -> int:
    return 128


@pytest.fixture
def transformer_model(trf_width, trf_depth) -> Model[List[Doc], List[TransformerData]]:
    return DummyTransformer(width=trf_width, depth=trf_depth)


@pytest.fixture
def docs() -> List[Doc]:
    vocab = Vocab()
    return [Doc(vocab, words=["hello", "one"]), Doc(vocab, words=["hi", "two"])]


@pytest.fixture
def transformer_batch(transformer_model, docs):
    return transformer_model.predict(docs).doc_data


def test_init_linear_no_dims():
    model = transformer_linear_v1()
    assert model.has_dim("nO") is None
    assert model.has_dim("nI") is None
    assert len(model.layers) == 1
    assert model.name == "trf-linear"


def test_init_linear_dims(output_width, trf_width):
    model = transformer_linear_v1(nO=output_width, nI=trf_width)
    model.initialize()
    assert model.get_dim("nO") == output_width
    assert model.get_dim("nI") == trf_width
    assert len(model.layers) == 1
    assert model.name == "trf-linear"


def test_linear_forward(output_width, trf_width, transformer_batch):
    model = transformer_linear_v1(nO=output_width, nI=transformer_batch[0].width)
    model.initialize()
    outputs, backprop = model.begin_update(transformer_batch)
    for y in outputs:
        assert y.width == output_width
    d_inputs = backprop(outputs)
    for dx in d_inputs:
        assert dx.width == trf_width
