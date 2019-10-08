import numpy
import pytest
from thinc.neural.optimizers import Adam
from spacy_transformers.activations import Activations, RaggedArray
from spacy_transformers.util import get_tokenizer, PIPES


@pytest.fixture
def tokenizer(name):
    return get_tokenizer(name).from_pretrained(name)


@pytest.fixture
def ids(tokenizer):
    text = "the cat sat on the mat"
    return numpy.array(tokenizer.encode(text), dtype=numpy.int_)


@pytest.fixture
def inputs(ids):
    return RaggedArray(ids, [len(ids)])


@pytest.fixture
def model(nlp):
    return nlp.get_pipe(PIPES.tok2vec).model._model


def test_wrapper_from_pretrained(name, model, inputs):
    outputs, backprop = model.begin_update(inputs)
    assert outputs.has_lh
    optimizer = Adam(model.ops, 0.001)
    d_outputs = Activations(outputs.lh, RaggedArray.blank())
    backprop(d_outputs, sgd=optimizer)
