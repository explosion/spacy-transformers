from spacy_pytorch_transformers.wrapper import PyTT_Wrapper
import pytorch_transformers as pytt
import numpy
import pytest

from thinc.neural.optimizers import Adam


@pytest.fixture
def name():
    return "bert-base-uncased"


@pytest.fixture()
def text():
    return "the cat sat on the mat"


@pytest.fixture
def tokenizer(name):
    return pytt.BertTokenizer.from_pretrained(name)


@pytest.fixture
def ids(tokenizer, text):
    return numpy.array(tokenizer.encode(text), dtype=numpy.int_)


def test_bert_wrapper_from_pretrained(name, ids):
    model = PyTT_Wrapper.from_pretrained(name)
    outputs, backprop = model.begin_update(ids.reshape((1, -1)))
    assert len(outputs) == 2
    assert outputs.last_hidden_state.shape == (1, 6, 768)
    assert outputs.pooler_output.shape == (1, 768)
    optimizer = Adam(model.ops, 0.001)
    backprop(outputs, sgd=optimizer)
