import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from .._align import get_alignment


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
        ([["ab", "cd"]], [["a", "bc" "d"]]),
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
        wp_word = "".join([flat_words2[int(j)] for j in align[i].data])
        if len(word) < len(wp_word):
            assert word in wp_word
        elif len(word) > len(wp_word):
            assert wp_word in word
        else:
            assert word == wp_word
