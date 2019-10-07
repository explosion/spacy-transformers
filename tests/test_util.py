import pytest
import numpy
from numpy.testing import assert_equal
from spacy_transformers.pipeline.wordpiecer import align_word_pieces
from spacy_transformers.util import pad_batch, batch_by_length, lengths2mask


@pytest.mark.parametrize(
    "spacy_tokens,wp_tokens,expected_alignment",
    [
        (["a"], ["a"], [[0]]),
        (["a", "b", "c"], ["a", "b", "c"], [[0], [1], [2]]),
        (["ab", "c"], ["a", "b", "c"], [[0, 1], [2]]),
        (["a", "b", "c"], ["ab", "c"], [[0], [0], [1]]),
        (["ab", "cd"], ["a", "bc", "d"], [[0, 1], [1, 2]]),
        (["abcd"], ["ab", "cd"], [[0, 1]]),
        ([], [], []),
        (
            ["it", "realllllllly", "sucks", "."],
            ["it", "real", "ll", "ll", "ll", "ly", "suck", "s", "."],
            [[0], [1, 2, 3, 4, 5], [6, 7], [8]],
        ),
        (["Well", ",", "i"], ["Well", ",", "", "i"], [[0], [1, 2], [2, 3]]),
    ],
)
def test_align_word_pieces(spacy_tokens, wp_tokens, expected_alignment):
    output = align_word_pieces(spacy_tokens, wp_tokens)
    assert output == expected_alignment


@pytest.mark.parametrize(
    "lengths,max_words,expected",
    [
        ([1, 2, 2, 4], 4, [[3], [2], [0, 1]]),
        ([1, 2, 2, 4], 3, [[3], [2], [1], [0]]),
        ([4, 2, 2, 1], 4, [[0], [2], [1, 3]]),
        ([4, 4, 2, 2, 1], 6, [[1], [0], [2, 3, 4]]),
        ([10, 7, 2, 2, 1], 10, [[0], [1], [2, 3, 4]]),
    ],
)
def test_batch_by_length(lengths, max_words, expected):
    seqs = ["a" * length for length in lengths]
    batches = batch_by_length(seqs, max_words)
    assert batches == expected


@pytest.mark.parametrize(
    "lengths,expected_shape",
    [
        ([1, 2], (2, 2)),
        ([1, 2, 3], (3, 3)),
        ([0, 1], (2, 1)),
        ([1], (1, 1)),
        ([1, 5, 2, 4], (4, 5)),
    ],
)
def test_pad_batch(lengths, expected_shape):
    seqs = [numpy.ones((length,), dtype="f") for length in lengths]
    padded = pad_batch(seqs)
    for i, seq in enumerate(seqs):
        padded[i][padded[i] < 0] = 0
        assert padded[i].sum() == seq.sum()
        assert_equal(padded[i, : len(seq)], seq)
    assert padded.shape == expected_shape


@pytest.mark.parametrize(
    "lengths,width,expected_shape",
    [
        ([1, 2], 5, (2, 2, 5)),
        ([1, 2, 3], 2, (3, 3, 2)),
        ([0, 1], 3, (2, 1, 3)),
        ([1], 4, (1, 1, 4)),
        ([1, 5, 2, 4], 6, (4, 5, 6)),
    ],
)
def test_pad_batch_2d(lengths, width, expected_shape):
    seqs = [numpy.ones((length, width), dtype="f") for length in lengths]
    padded = pad_batch(seqs)
    for i, seq in enumerate(seqs):
        padded[i][padded[i] < 0] = 0
        assert padded[i].sum() == seq.sum()
        assert_equal(padded[i, : len(seq)], seq)
    assert padded.shape == expected_shape


@pytest.mark.parametrize("lengths", [[1], [1, 2], [5, 3, 1]])
def test_lengths2mask(lengths):
    mask = lengths2mask(lengths)
    assert mask.ndim == 1
    assert mask.sum() == sum(lengths)
    assert mask.size == max(lengths) * len(lengths)
