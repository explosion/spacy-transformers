import pytest
from spacy_pytorch_transformers import PyTT_WordPiecer
from spacy_pytorch_transformers.util import is_special_token, get_pytt_tokenizer
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
    assert "".join(cleaned_words) == "".join(words)


def test_tokenizers_to_from_bytes(name):
    text = "hello world"
    tokenizer_cls = get_pytt_tokenizer(name)
    tokenizer = tokenizer_cls.from_pretrained(name)
    doc = tokenizer.tokenize(text)
    assert isinstance(doc, list) and len(doc)
    bytes_data = tokenizer.to_bytes(name)
    new_tokenizer = tokenizer_cls.blank().from_bytes(bytes_data)
    new_doc = new_tokenizer.tokenize(text)
    assert isinstance(new_doc, list) and len(new_doc)
    assert doc == new_doc
