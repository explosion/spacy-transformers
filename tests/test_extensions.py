import pytest
from spacy_transformers import TransformersWordPiecer
from spacy_transformers.util import ATTRS
from spacy.tokens import Doc
from spacy.vocab import Vocab


@pytest.fixture
def vocab():
    return Vocab()


@pytest.fixture
def wordpiecer(name):
    return TransformersWordPiecer.from_pretrained(vocab, trf_name=name)


def test_alignment_extension_attr(vocab):
    doc = Doc(vocab, words=["hello", "world", "test"])
    doc._.set(ATTRS.alignment, [[1, 2], [3, 4], [5, 6]])
    assert doc[0]._.get(ATTRS.alignment) == [1, 2]
    assert doc[1]._.get(ATTRS.alignment) == [3, 4]
    assert doc[2]._.get(ATTRS.alignment) == [5, 6]
    assert doc[0:2]._.get(ATTRS.alignment) == [[1, 2], [3, 4]]
    assert doc[1:3]._.get(ATTRS.alignment) == [[3, 4], [5, 6]]


def test_wp_span_extension_attr(name, vocab, wordpiecer):
    if name == "gpt2":
        return
    doc = Doc(vocab, words=["hello", "world"])
    for w in doc[1:]:
        w.is_sent_start = False
    doc = wordpiecer(doc)
    assert len(doc._.get(ATTRS.word_pieces)) == 4
    assert doc[0]._.get(ATTRS.start) == 0
    assert doc[-1]._.get(ATTRS.end) == 3
    assert len(doc[:]._.get(ATTRS.word_pieces)) == 4
