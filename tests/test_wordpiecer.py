import pytest
from spacy_transformers import TransformersWordPiecer
from spacy_transformers.util import is_special_token, get_tokenizer, ATTRS
from spacy.vocab import Vocab
from spacy.tokens import Doc


@pytest.fixture(scope="session")
def wp(name):
    return TransformersWordPiecer.from_pretrained(Vocab(), trf_name=name)


def test_wordpiecer(wp):
    words = ["hello", "world", "this", "is", "a", "test"]
    doc = Doc(wp.vocab, words=words)
    doc[0].is_sent_start = True
    doc[1].is_sent_start = False
    doc = wp(doc)
    cleaned_words = [wp.model.clean_wp_token(t) for t in doc._.get(ATTRS.word_pieces_)]
    cleaned_words = [w for w in cleaned_words if not is_special_token(w)]
    assert "".join(cleaned_words) == "".join(words)


def test_xlnet_weird_align(name, wp):
    if "xlnet" not in name.lower():
        return True
    text = "Well, i rented this movie and found out it realllllllly sucks."
    spacy_tokens = [
        "Well",
        ",",
        "i",
        "rented",
        "this",
        "movie",
        "and",
        "found",
        "out",
        "it",
        "realllllllly",
        "sucks",
        ".",
    ]
    spaces = [True] * len(spacy_tokens)
    spaces[0] = False
    spaces[-2] = False
    spaces[-1] = False
    doc = Doc(wp.vocab, words=spacy_tokens, spaces=spaces)
    doc[1].is_sent_start = False
    assert doc.text == text
    doc = wp(doc)
    assert doc._.get(ATTRS.word_pieces_)[-2] == "</s>"
    assert doc._.get(ATTRS.word_pieces_)[-1] == "<cls>"


def test_tokenizers_to_from_bytes(name):
    text = "hello world"
    tokenizer_cls = get_tokenizer(name)
    tokenizer = tokenizer_cls.from_pretrained(name)
    doc = tokenizer.tokenize(text)
    assert isinstance(doc, list) and len(doc)
    bytes_data = tokenizer.to_bytes(name)
    new_tokenizer = tokenizer_cls.blank().from_bytes(bytes_data)
    new_doc = new_tokenizer.tokenize(text)
    assert isinstance(new_doc, list) and len(new_doc)
    assert doc == new_doc
