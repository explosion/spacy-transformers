import pytest
from spacy.vocab import Vocab
from spacy.tokens import Doc
from thinc.api import Model
from ..pipeline import Transformer, AnnotationSetter
from .util import DummyTransformer
from ..types import TransformerOutput
from ..extensions import install_extensions


@pytest.fixture
def vocab():
    return Vocab()

@pytest.fixture
def docs(vocab):
    return [Doc(vocab, words=["hello", "world"]), Doc(vocab, words=["this", "is", "another"])]


@pytest.fixture
def component(vocab):
    return Transformer(Vocab(), DummyTransformer())

def test_init(component):
    assert isinstance(component.vocab, Vocab)
    assert isinstance(component.model, Model)
    assert isinstance(component.annotation_setter, AnnotationSetter)
    assert component.listeners == []
    assert component.cfg == {}


def test_predict(component, docs):
    trf_data = component.predict(docs)
    assert isinstance(trf_data, TransformerOutput)
    assert len(trf_data.tensors) == component.model.layers[0].attrs["depth"]
    n_tokens = trf_data.tokens.input_ids.shape[1]
    width = component.model.layers[0].attrs["width"]
    assert trf_data.arrays[-1].shape == (len(docs), n_tokens, width)


def test_set_annotations(component, docs):
    install_extensions()
    trf_data = component.predict(docs)
    component.set_annotations(docs, trf_data)
    for doc in docs:
        assert doc._.trf_data is trf_data
    for i, span in enumerate(trf_data.spans):
        assert span._.trf_row == i
    for doc in docs:
        for token in doc:
            assert isinstance(token._.trf_alignment, list)


def test_listeners(component, docs):
    width = component.model.layers[0].attrs["width"]
    docs = list(component.pipe(docs)) 
    for listener in component.listeners:
        assert listener.verify_inputs(docs)
