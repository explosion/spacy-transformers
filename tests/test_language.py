from spacy_pytorch_transformers import PyTT_Language
from spacy_pytorch_transformers import about
from spacy.attrs import LANG


def test_language_init():
    meta = {"lang": "en", "name": "test", "pipeline": []}
    nlp = PyTT_Language(meta=meta, pytt_name="bert-base-uncased")
    assert nlp.lang == "en"
    assert nlp.meta["lang"] == "en"
    assert nlp.meta["lang_factory"] == PyTT_Language.lang_factory_name
    assert nlp.vocab.lang == "en"
    # Make sure we really have the EnglishDefaults here
    assert nlp.Defaults.lex_attr_getters[LANG](None) == "en"
    # Test requirements
    package = f"{about.__title__}>={about.__version__}"
    assert package in nlp.meta["requirements"]
