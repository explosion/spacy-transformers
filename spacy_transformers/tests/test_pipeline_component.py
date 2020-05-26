import pytest
from spacy.vocab import Vocab
from spacy.tokens import Doc
from thinc.api import Model

from .util import DummyTransformer
from ..pipeline_component import Transformer
from ..data_classes import TransformerData, FullTransformerBatch


@pytest.fixture
def vocab():
    return Vocab()


@pytest.fixture
def docs(vocab):
    return [
        Doc(vocab, words=["hello", "world"]),
        Doc(vocab, words=["this", "is", "another"]),
    ]


@pytest.fixture
def component(vocab):
    return Transformer(Vocab(), DummyTransformer())


def test_init(component):
    assert isinstance(component.vocab, Vocab)
    assert isinstance(component.model, Model)
    assert hasattr(component.annotation_setter, "__call__")
    assert component.listeners == []
    assert component.cfg == {"max_batch_size": 8}


def test_predict(component, docs):
    trf_data = component.predict(docs)
    assert isinstance(trf_data, FullTransformerBatch)
    assert len(trf_data.tensors) == component.model.layers[0].attrs["depth"]
    n_tokens = trf_data.tokens["input_ids"].shape[1]
    width = component.model.layers[0].attrs["width"]
    assert trf_data.tensors[-1].shape == (len(docs), n_tokens, width)


def test_set_annotations(component, docs):
    trf_data = component.predict(docs)
    component.set_annotations(docs, trf_data)
    for doc in docs:
        assert isinstance(doc._.trf_data, TransformerData)


def test_listeners(component, docs):
    docs = list(component.pipe(docs))
    for listener in component.listeners:
        assert listener.verify_inputs(docs)
