import pytest
from .._align import BatchAlignment


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
        ([["ab", "cd"]], [["a", "bc" "d"]]),
    ],
)
def test_alignments_match(words1, words2):
    align = BatchAlignment.from_strings(words1, words2)
    flat_words1, flat_words2 = flatten_strings(words1, words2)
    print("Flat words1", flat_words1)
    print("Flat words2", flat_words2)
    for i, word in enumerate(flat_words1):
        wp_word = "".join([flat_words2[int(j)] for j in align.wp2tok[i].data])
        if len(word) < len(wp_word):
            assert word in wp_word
        elif len(word) > len(wp_word):
            assert wp_word in word
        else:
            assert word == wp_word
