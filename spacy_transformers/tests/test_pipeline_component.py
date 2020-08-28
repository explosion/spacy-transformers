import pytest
from spacy.gold import Example
from spacy.lang.en import English
from spacy.vocab import Vocab
from spacy.tokens import Doc
from thinc.api import Model

from .util import DummyTransformer
from ..annotation_setters import configure_trfdata_setter
from ..layers import TransformerListener
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
    return Transformer(Vocab(), DummyTransformer(), configure_trfdata_setter())


def test_init(component):
    assert isinstance(component.vocab, Vocab)
    assert isinstance(component.model, Model)
    assert hasattr(component.annotation_setter, "__call__")
    assert component.listeners == []
    assert component.cfg == {"max_batch_items": 4096}


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


def test_pipeline(vocab, docs):
    nlp = English(vocab)
    nlp.add_pipe("transformer")
    nlp.add_pipe(
        "tagger",
        config={
            "model": {
                "tok2vec": {
                    "@architectures": "spacy-transformers.TransformerListener.v1",
                    "pooling": {"@layers": "reduce_mean.v1"},
                }
            }
        },
    )
    assert nlp.pipe_names == ['transformer', 'tagger']
    tagger = nlp.get_pipe("tagger")
    assert isinstance(tagger.model.get_ref("tok2vec").layers[0], TransformerListener)
    examples = [Example.from_dict(d, {}) for d in docs]
    nlp.begin_training(lambda: examples)

    nlp.disable_pipes("transformer")
    with pytest.raises(AssertionError):
        nlp("This is bound to go wrong")
