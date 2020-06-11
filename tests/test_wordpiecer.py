import pytest
from spacy_transformers import TransformersWordPiecer
from spacy_transformers.util import is_special_token, get_tokenizer, ATTRS
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.pipeline import Sentencizer


@pytest.fixture(scope="session")
def wp(name):
    return TransformersWordPiecer.from_pretrained(Vocab(), trf_name=name)


@pytest.fixture(scope="session")
def sentencizer():
    return Sentencizer()


def test_wordpiecer(wp):
    words = ["hello", "world", "this", "is", "a", "test"]
    doc = Doc(wp.vocab, words=words)
    doc[0].is_sent_start = True
    doc[1].is_sent_start = False
    doc = wp(doc)
    cleaned_words = [wp.model.clean_wp_token(t) for t in doc._.get(ATTRS.word_pieces_)]
    cleaned_words = [w for w in cleaned_words if not is_special_token(w)]
    assert "".join(cleaned_words) == "".join(words)


@pytest.mark.parametrize(
    "words,target_name,expected_align",
    [
        (
            ["hello", "world", "this", "is", "a", "teest"],
            "bert-base-uncased",
            [[1], [2], [3], [4], [5], [6, 7]],
        ),
        (
            ["hello", "world", "this", "is", "a", "teest"],
            "xlnet-base-cased",
            [[0], [1], [2], [3], [4], [5, 6]],
        ),
        (["å\taa", ".", "が\nπ"], "bert-base-uncased", [[1, 2], [3], [6, 7]]),
        (["å\taa", ".", "が\nπ"], "xlnet-base-cased", [[0, 1, 2], [4], [7, 8, 10]]),
        (["\u3099"], "bert-base-uncased", [[]]),
        (["I.\n\n\n\n\n"], "bert-base-uncased", [[1, 2]]),
        # max length of 512 minus 2 special tokens -> 510 aligned tokens,
        # remaining truncated per sentence
        (
            ["x"] * 599 + ["."] + ["y"] * 600,
            "bert-base-uncased",
            [[x] for x in range(1, 511)]
            + [[]] * 90
            + [[x] for x in range(513, 1023)]
            + [[]] * 90,
        ),
    ],
)
def test_align(wp, sentencizer, name, words, target_name, expected_align):
    if name != target_name:
        pytest.skip()
    doc = Doc(wp.vocab, words=words)
    doc = sentencizer(doc)
    doc = wp(doc)
    assert doc._.get(ATTRS.alignment) == expected_align


def test_xlnet_weird_align(name, wp):
    if "xlnet" not in name.lower():
        pytest.skip()
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
