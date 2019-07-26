import pytest
from spacy_pytorch_transformers.util import align_word_pieces, pad_batch, batch_by_length


@pytest.mark.parametrize("spacy_tokens,wp_tokens,expected_alignment", [
    (["a"], ["a"], [[0]]),
    (["a", "b", "c"], ["a", "b", "c"], [[0], [1], [2]]),
    (["ab", "c"], ["a", "b", "c"], [[0, 1], [2]]),
    (["a", "b", "c"], ["ab", "c"], [[0], [0], [1]]),
    (["ab", "cd"], ["a", "bc", "d"], [[0, 1], [1, 2]]),
    (["abcd"], ["ab", "##cd"], [[0, 1]]),
]) 
def test_align_word_pieces(spacy_tokens, wp_tokens, expected_alignment):
    output = align_word_pieces(spacy_tokens, wp_tokens)
    assert output == expected_alignment

