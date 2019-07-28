import pytest
from numpy.testing import assert_equal
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer
from spacy_pytorch_transformers import PyTT_TokenVectorEncoder, about
from spacy.attrs import LANG

from .util import make_tempdir, is_valid_tensor


@pytest.fixture(scope="session")
def name():
    return "bert-base-uncased"


@pytest.fixture(scope="session")
def nlp(name):
    cfg = {"batch_by_length": True}
    pytt_nlp = PyTT_Language(pytt_name=name)
    wordpiecer = PyTT_WordPiecer.from_pretrained(pytt_nlp.vocab, pytt_name=name)
    tok2vec = PyTT_TokenVectorEncoder.from_pretrained(pytt_nlp.vocab, name=name)
    pytt_nlp.add_pipe(wordpiecer)
    pytt_nlp.add_pipe(tok2vec)
    return pytt_nlp


def test_language_init(name):
    meta = {"lang": "en", "name": "test", "pipeline": []}
    nlp = PyTT_Language(meta=meta, pytt_name=name)
    assert nlp.lang == "en"
    assert nlp.meta["lang"] == "en"
    assert nlp.meta["lang_factory"] == PyTT_Language.lang_factory_name
    assert nlp.vocab.lang == "en"
    # Make sure we really have the EnglishDefaults here
    assert nlp.Defaults.lex_attr_getters[LANG](None) == "en"
    # Test requirements
    package = f"{about.__title__}>={about.__version__}"
    assert package in nlp.meta["requirements"]


def test_language_run(nlp):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)


def test_language_to_from_bytes(nlp, name):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)
    bytes_data = nlp.to_bytes()
    new_nlp = PyTT_Language()
    wordpiecer = PyTT_WordPiecer(new_nlp.vocab, pytt_name=name)
    tok2vec = PyTT_TokenVectorEncoder(new_nlp.vocab, name)
    new_nlp.add_pipe(wordpiecer)
    new_nlp.add_pipe(tok2vec)
    new_nlp.from_bytes(bytes_data)
    new_doc = new_nlp("hello world")
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)


def test_language_to_from_disk(nlp, name):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)
    with make_tempdir() as tempdir:
        nlp.to_disk(tempdir)
        new_nlp = PyTT_Language()
        wordpiecer = PyTT_WordPiecer(new_nlp.vocab, pytt_name=name)
        tok2vec = PyTT_TokenVectorEncoder(new_nlp.vocab, name)
        new_nlp.add_pipe(wordpiecer)
        new_nlp.add_pipe(tok2vec)
        new_nlp.from_disk(tempdir)
    assert new_nlp.pipe_names == nlp.pipe_names
    new_doc = new_nlp("hello world")
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)
