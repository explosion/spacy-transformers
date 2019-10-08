import pytest
from numpy.testing import assert_equal
from spacy_transformers import TransformersLanguage, TransformersWordPiecer
from spacy_transformers import TransformersTok2Vec, pkg_meta
from spacy_transformers.util import ATTRS
from spacy.attrs import LANG

from .util import make_tempdir, is_valid_tensor


def test_language_init(name):
    meta = {"lang": "en", "name": "test", "pipeline": []}
    nlp = TransformersLanguage(meta=meta, trf_name=name)
    assert nlp.lang == "en"
    assert nlp.meta["lang"] == "en"
    assert nlp.meta["lang_factory"] == TransformersLanguage.lang_factory_name
    assert nlp.vocab.lang == "en"
    # Make sure we really have the EnglishDefaults here
    assert nlp.Defaults.lex_attr_getters[LANG](None) == "en"
    # Test requirements
    package = f"{pkg_meta['title']}>={pkg_meta['version']}"
    assert package in nlp.meta["requirements"]


def test_language_run(nlp):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)


def test_language_wordpiece_to_from_bytes(name):
    nlp = TransformersLanguage()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    wordpiecer = TransformersWordPiecer.from_pretrained(nlp.vocab, trf_name=name)
    nlp.add_pipe(wordpiecer)
    doc = nlp("hello world")
    assert doc._.get(ATTRS.word_pieces) is not None
    nlp2 = TransformersLanguage()
    nlp2.add_pipe(nlp.create_pipe("sentencizer"))
    nlp2.add_pipe(TransformersWordPiecer(nlp2.vocab))
    with pytest.raises(ValueError):
        new_doc = nlp2("hello world")
    nlp2.from_bytes(nlp.to_bytes())
    new_doc = nlp2("hello world")
    assert new_doc._.get(ATTRS.word_pieces) is not None


def test_language_wordpiece_tok2vec_to_from_bytes(nlp, name):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)
    nlp2 = TransformersLanguage()
    nlp2.add_pipe(nlp2.create_pipe("sentencizer"))
    nlp2.add_pipe(TransformersWordPiecer(nlp.vocab))
    nlp2.add_pipe(TransformersTok2Vec(nlp.vocab))
    with pytest.raises(ValueError):
        new_doc = nlp2("hello world")
    nlp2.from_bytes(nlp.to_bytes())
    new_doc = nlp2("hello world")
    assert is_valid_tensor(new_doc.tensor)
    assert new_doc._.get(ATTRS.word_pieces) is not None


def test_language_to_from_disk(nlp, name):
    doc = nlp("hello world")
    assert is_valid_tensor(doc.tensor)
    with make_tempdir() as tempdir:
        nlp.to_disk(tempdir)
        new_nlp = TransformersLanguage()
        new_nlp.add_pipe(new_nlp.create_pipe("sentencizer"))
        wordpiecer = TransformersWordPiecer(new_nlp.vocab, trf_name=name)
        tok2vec = TransformersTok2Vec(new_nlp.vocab, trf_name=name)
        new_nlp.add_pipe(wordpiecer)
        new_nlp.add_pipe(tok2vec)
        new_nlp.from_disk(tempdir)
    assert new_nlp.pipe_names == nlp.pipe_names
    new_doc = new_nlp("hello world")
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)
