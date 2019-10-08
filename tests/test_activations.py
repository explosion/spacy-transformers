import pytest
import numpy
from spacy_transformers.activations import Activations, RaggedArray


def test_act_blank():
    acts = Activations.blank()
    assert acts.lh.data.size == 0
    assert acts.lh.lengths == []
    assert acts.po.data.size == 0
    assert acts.po.lengths == []
    assert not acts.has_lh
    assert not acts.has_po


@pytest.mark.parametrize(
    "extra_dims,lengths,pad_to,expected_shape",
    [
        ((), [1], 1, (1, 1)),
        ((), [1, 2], -1, (2, 2)),
        ((3,), [1, 2], -1, (2, 2, 3)),
        ((3,), [1, 2, 5], -1, (3, 5, 3)),
        ((3,), [1, 2, 5], 4, (3, 5, 3)),
    ],
)
def test_ragged_to_padded(extra_dims, lengths, pad_to, expected_shape):
    arr = RaggedArray(numpy.ones((sum(lengths),) + extra_dims), lengths)
    if pad_to > 1 and pad_to < max(lengths):
        with pytest.raises(ValueError):
            padded = arr.to_padded(to=pad_to)
    else:
        padded = arr.to_padded(to=pad_to)
        assert padded.shape == expected_shape


@pytest.mark.parametrize(
    "shape,lengths,expected_shape",
    [
        ((1, 2), [3], (3,)),
        ((2, 3), [3, 2], (5,)),
        ((2, 4, 4), [4, 2], (6, 4)),
        ((4, 5, 2), [5, 2, 1, 3], (11, 2)),
    ],
)
def test_ragged_from_padded(shape, lengths, expected_shape):
    padded = numpy.ones(shape, dtype="i")
    for i, length in enumerate(lengths):
        padded[i, length:] = 0
    ragged = RaggedArray.from_padded(padded, lengths)
    assert ragged.data.shape == expected_shape
    assert ragged.data.sum() == padded.sum()


@pytest.mark.parametrize(
    "shape,lengths,expected_shape",
    [
        ((1, 2), [3], (3,)),
        ((2, 2), [3, 2], (5,)),
        ((2, 2, 4), [3, 2], (5, 4)),
        ((4, 2, 2), [5, 2, 1, 3], (11, 2)),
    ],
)
def test_ragged_from_truncated(shape, lengths, expected_shape):
    truncated = numpy.ones(shape, dtype="i")
    for i, length in enumerate(lengths):
        truncated[i, length:] = 0
    ragged = RaggedArray.from_truncated(truncated, lengths)
    assert ragged.data.shape == expected_shape
    assert ragged.data.sum() == truncated.sum()
