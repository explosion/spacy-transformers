import pytest
from typing import List
import numpy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from thinc.api import NumpyOps
from thinc.types import Ragged
from ..align import get_alignment, apply_alignment
from ..align import get_span2wp_from_offset_mapping


def get_ragged(ops, nested: List[List[int]]):
    nested = [ops.asarray(x) for x in nested]
    return Ragged(ops.flatten(nested), ops.asarray([len(x) for x in nested]))


def get_spans(word_seqs):
    vocab = Vocab()
    docs = [Doc(vocab, words=words) for words in word_seqs]
    return [doc[:] for doc in docs]


def flatten_strings(words1, words2):
    flat1 = []
    flat2 = []
    for seq in words1:
        flat1.extend(seq)
    stride = max((len(seq) for seq in words2), default=0)
    for seq in words2:
        flat2.extend(seq)
        flat2.extend([""] * (stride - len(seq)))
    return flat1, flat2


@pytest.mark.parametrize(
    "words1,words2",
    [
        ([["a", "b"]], [["a", "b"]]),
        ([["ab"]], [["a", "b"]]),
        ([["a", "b"]], [["ab"]]),
        ([["ab", "c"]], [["a", "bc"]]),
        ([["ab", "cd"]], [["a", "bc", "d"]]),
    ],
)
def test_alignments_match(words1, words2):
    spans = get_spans(words1)
    align = get_alignment(spans, words2)
    unique_tokens = set()
    for span in spans:
        for token in span:
            unique_tokens.add((id(token.doc), token.idx))
    assert len(unique_tokens) == align.lengths.shape[0]
    flat_words1, flat_words2 = flatten_strings(words1, words2)
    for i, word in enumerate(flat_words1):
        wp_word = "".join([flat_words2[int(j[0])] for j in align[i].data])
        if len(word) < len(wp_word):
            assert word in wp_word
        elif len(word) > len(wp_word):
            assert wp_word in word
        else:
            assert word == wp_word


@pytest.mark.parametrize(
    "nested_align,X_cols",
    [
        ([[0, 1, 2], [3], [4]], 4),
        ([[], [1], [1], [2]], 2),
        ([[0, 1], [1, 2], [], [4]], 2),
    ],
)
def test_apply_alignment(nested_align, X_cols):
    ops = NumpyOps()
    align = get_ragged(ops, nested_align)
    X_shape = (align.data.max() + 1, X_cols)
    X = ops.alloc2f(*X_shape)
    Y, get_dX = apply_alignment(ops, align, X)
    assert isinstance(Y, Ragged)
    assert Y.data.shape[0] == align.data.shape[0]
    assert Y.lengths.shape[0] == len(nested_align)
    dX = get_dX(Y)
    assert dX.shape == X.shape


@pytest.mark.parametrize(
    # fmt: off
    # roberta-base offset_mapping and expected alignment
    "words,offset_mapping,alignment",
    [
        (
            ["Áaaa"],
            numpy.asarray([(0, 0), (0, 1), (0, 1), (1, 4), (0, 0)], dtype="i"),
            [[1, 2, 3]],
        ),
        (
            ["INGG", "á", "aäa"],
            numpy.asarray([(0, 0), (0, 3), (3, 4), (5, 6), (5, 6), (7, 8), (8, 9), (9, 10), (0, 0)], dtype="i"),
            [[1, 2], [3, 4], [5, 6, 7]],
        ),
    ],
    # fmt: on
)
def test_offset_alignment(words, offset_mapping, alignment):
    spans = get_spans([words])
    result = get_span2wp_from_offset_mapping(spans[0], offset_mapping)
    assert all(sorted(r) == a for r, a in zip(result, alignment))
