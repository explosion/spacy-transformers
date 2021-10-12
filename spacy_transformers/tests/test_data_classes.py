import pytest
import numpy
from numpy.testing import assert_equal
from spacy_transformers.data_classes import WordpieceBatch


@pytest.fixture
def wordpieces():
    strings = [["some", "random", "strings"], ["are"], ["added", "here"]]
    shape = (len(strings), max(len(seq) for seq in strings))
    wordpieces = WordpieceBatch(
        strings=strings,
        input_ids=numpy.zeros(shape, dtype="i"),
        token_type_ids=numpy.zeros(shape, dtype="i"),
        attention_mask=numpy.zeros((shape[0], shape[1]), dtype="bool"),
        lengths=[len(seq) for seq in strings],
    )
    return wordpieces


def test_wordpieces_IO(wordpieces):
    wp_dict = wordpieces.to_dict()
    wordpieces_2 = WordpieceBatch.empty().from_dict(wp_dict)
    for key, value in wordpieces_2.to_dict().items():
        assert_equal(value, wp_dict[key])
