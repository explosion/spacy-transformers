from spacy_pytorch_transformers.wrapper import PyTT_Wrapper
from spacy_pytorch_transformers.util import get_pytt_tokenizer
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


def test_wrapper_from_pretrained(name, ids):
    model = PyTT_Wrapper.from_pretrained(name)
    outputs, backprop = model.begin_update(ids.reshape((1, -1)))
    assert outputs.has_last_hidden_state
    if outputs.has_pooler_output:
        assert hasattr(outputs.pooler_output, "shape")
    optimizer = Adam(model.ops, 0.001)
    backprop(outputs.last_hidden_state, sgd=optimizer)
