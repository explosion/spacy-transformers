import pytest
import numpy
from spacy_transformers.truncate import _truncate_tokens, _truncate_alignment
from thinc.types import Ragged
from thinc.api import NumpyOps


@pytest.fixture
def sequences():
    # Each sequence is a list of tokens, and each token is a number of wordpieces
    return [
        [1, 3, 1],    # So 5 wordpieces this sequence
        [3, 7, 1, 1], # 12
        [1],          # 1
        [20, 1],      # 21
    ]

@pytest.fixture
def shape(sequences):
    # Get the shape of the input_ids, which includes the padding.
    maximum = max(sum(lengths) for lengths in sequences)
    return (len(sequences), maximum)


@pytest.fixture
def seq_lengths(sequences):
    return numpy.array([sum(seq) for seq in sequences], dtype="i")


@pytest.fixture
def align(sequences):
    lengths = []
    indices = []
    offset = 0
    for seq in sequences:
        for token_length in seq:
            lengths.append(token_length)
            indices.extend(i+offset for i in range(token_length))
            offset += token_length
    return Ragged(numpy.array(indices, dtype="i"), numpy.array(lengths, dtype="i"))


@pytest.fixture
def max_length():
    return 6


@pytest.fixture
def mask_from_end(shape, max_length):
    n_seq, length = shape
    bools = [
        numpy.array([(i+1) < max_length for i in range(length)], dtype="bool")
        for _ in range(n_seq)
    ]
    return numpy.concatenate(bools)


def test_truncate_alignment_from_end(sequences, max_length, align, mask_from_end):
    #print("Max length", max_length)
    #print("Sequences", sequences)
    #print("Mask", mask_from_end)
    ops = NumpyOps()
    truncated = _truncate_alignment(align, mask_from_end)
    #print(truncated.dataXd.shape, truncated.lengths.sum())
    #print("Before", list(map(list, ops.unflatten(align.dataXd, align.lengths))))
    #print("After", list(map(list, ops.unflatten(truncated.dataXd, truncated.lengths))))
    # Check that the number of tokens hasn't changed. We still need to have
    # alignment for every token.
    assert truncated.lengths.shape[0] == align.lengths.shape[0]
    start = 0
    for i, seq in enumerate(sequences):
        end = start + len(seq)
        # Get the alignment for this sequence of tokens. Each length in the
        # alignment indicates the number of wordpiece tokens, so we need to
        # check that the sum of the lengths doesn't exceed the maximum.
        wp_indices = truncated[start:end]
        assert wp_indices.lengths.sum() < max_length
        # We're truncating from the end, so we shouldn't see different values
        # except at the end of the sequence.
        seen_zero = False
        before = align[start:end]
        for length_now, length_before in zip(wp_indices.lengths, before.lengths):
            if seen_zero:
                assert length_now == 0
            elif length_now == 0:
                seen_zero = True
            else:
                length_now == length_before
