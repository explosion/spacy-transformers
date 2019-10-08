import pytest
import numpy
from thinc.api import layerize
from spacy_transformers.model_registry import with_length_batching
from spacy_transformers.activations import Activations, RaggedArray


def make_activations(inputs, width):
    """Make an Activations object with the right sizing for an input."""
    lh = numpy.ones((inputs.data.shape[0], width), dtype="f")
    po = numpy.ones((len(inputs.lengths), width), dtype="f")
    return Activations(
        RaggedArray(lh, inputs.lengths), RaggedArray(po, [1 for _ in inputs.lengths])
    )


def create_dummy_model(width):
    record_fwd = []
    record_bwd = []

    @layerize
    def dummy_model(inputs, drop=0.0):
        def dummy_model_backward(d_outputs, sgd=None):
            record_bwd.append(d_outputs)
            return None

        record_fwd.append(inputs)
        return make_activations(inputs, width), dummy_model_backward

    return dummy_model, record_fwd, record_bwd


@pytest.mark.parametrize(
    "max_words,lengths,expected_batches",
    [(10, [4, 2, 2, 8, 2, 10], [[10], [8], [4], [2, 2, 2]]), (500, [9, 5], [[9, 5]])],
)
def test_with_length_batching(max_words, lengths, expected_batches, width=12):
    inputs = RaggedArray(numpy.arange(sum(lengths), dtype="i"), lengths)
    # The record_fwd and record_bwd variables will record the calls into the
    # forward and backward passes, respectively. We can use this to check that
    # the batching was done correctly.
    dummy, record_fwd, record_bwd = create_dummy_model(width)
    batched_dummy = with_length_batching(dummy, max_words)
    outputs, backprop = batched_dummy.begin_update(inputs)
    assert len(record_fwd) == len(expected_batches)
    assert [b.lengths for b in record_fwd] == expected_batches
    assert outputs.lh.data.shape == (inputs.data.shape[0], width)
    assert outputs.po.data.shape == (len(inputs.lengths), width)
    none = backprop(outputs)
    assert none is None
    assert len(record_bwd) == len(expected_batches)
