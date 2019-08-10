import pytest
from numpy.testing import assert_equal
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer
from spacy_pytorch_transformers import PyTT_TokenVectorEncoder, about
from spacy.attrs import LANG

from .util import make_tempdir, is_valid_tensor


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


def test_language_wordpiece_to_from_bytes(name):
    nlp = PyTT_Language()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    wordpiecer = PyTT_WordPiecer.from_pretrained(nlp.vocab, pytt_name=name)
    nlp.add_pipe(wordpiecer)
    doc = nlp("hello world")
    assert doc._.pytt_word_pieces is not None
    nlp2 = PyTT_Language()
    nlp2.add_pipe(nlp.create_pipe("sentencizer"))
    nlp2.add_pipe(PyTT_WordPiecer(nlp2.vocab))
    with pytest.raises(ValueError):
        new_doc = nlp2("hello world")
    nlp2.from_bytes(nlp.to_bytes())
    new_doc = nlp2("hello world")
    assert new_doc._.pytt_word_pieces is not None


def test_language_wordpiece_tok2vec_to_from_bytes(nlp, name):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)
    nlp2 = PyTT_Language()
    nlp2.add_pipe(nlp2.create_pipe("sentencizer"))
    nlp2.add_pipe(PyTT_WordPiecer(nlp.vocab))
    nlp2.add_pipe(PyTT_TokenVectorEncoder(nlp.vocab))
    with pytest.raises(ValueError):
        new_doc = nlp2("hello world")
    nlp2.from_bytes(nlp.to_bytes())
    new_doc = nlp2("hello world")
    assert is_valid_tensor(new_doc.tensor)
    assert new_doc._.pytt_word_pieces is not None


def test_language_to_from_disk(nlp, name):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)
    with make_tempdir() as tempdir:
        nlp.to_disk(tempdir)
        new_nlp = PyTT_Language()
        new_nlp.add_pipe(new_nlp.create_pipe("sentencizer"))
        wordpiecer = PyTT_WordPiecer(new_nlp.vocab, pytt_name=name)
        tok2vec = PyTT_TokenVectorEncoder(new_nlp.vocab, pytt_name=name)
        new_nlp.add_pipe(wordpiecer)
        new_nlp.add_pipe(tok2vec)
        new_nlp.from_disk(tempdir)
    assert new_nlp.pipe_names == nlp.pipe_names
    new_doc = new_nlp("hello world")
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)
