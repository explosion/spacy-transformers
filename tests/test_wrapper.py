from spacy_pytorch_transformers.util import get_pytt_tokenizer, Activations
import numpy
import pytest

from thinc.neural.optimizers import Adam


@pytest.fixture
def tokenizer(name):
    return get_pytt_tokenizer(name).from_pretrained(name)


@pytest.fixture
def ids(tokenizer):
    text = "the cat sat on the mat"
    return numpy.array(tokenizer.encode(text), dtype=numpy.int_)


@pytest.fixture
def model(nlp):
    return nlp.get_pipe("pytt_tok2vec").model._model


def test_wrapper_from_pretrained(name, model, ids):
    outputs, backprop = model.begin_update(ids.reshape((1, -1)))
    assert outputs.has_lh
    if outputs.has_po:
        assert hasattr(outputs.po[0], "shape")
    optimizer = Adam(model.ops, 0.001)
    d_outputs = Activations(outputs.lh, [], [], [], is_grad=True)
    backprop(d_outputs, sgd=optimizer)
