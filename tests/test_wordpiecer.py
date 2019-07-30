import pytest
from spacy_pytorch_transformers import PyTT_WordPiecer
from spacy_pytorch_transformers.util import is_special_token
from spacy.vocab import Vocab
from spacy.tokens import Doc


@pytest.fixture(scope="session")
def wp(name):
    return PyTT_WordPiecer.from_pretrained(Vocab(), pytt_name=name)


def test_wordpiecer(wp):
    words = ["hello", "world", "this", "is", "a", "test"]
    doc = Doc(wp.vocab, words=words)
    doc[0].is_sent_start = True
    doc[1].is_sent_start = False
    doc = wp(doc)
    cleaned_words = wp.model.clean_wp_tokens(doc._.pytt_word_pieces_)
    cleaned_words = [w for w in cleaned_words if not is_special_token(w)]
    assert cleaned_words == words
