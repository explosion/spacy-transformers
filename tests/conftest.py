import pytest
from spacy_transformers import TransformersLanguage, TransformersWordPiecer
from spacy_transformers import TransformersTok2Vec

MODEL_NAMES = [
    "bert-base-uncased",
    "gpt2",
    "xlnet-base-cased",
    "sshleifer/tiny-distilbert-base-cased",
]


@pytest.fixture(scope="session", params=MODEL_NAMES)
def name(request):
    return request.param


@pytest.fixture(scope="session")
def nlp(name):
    p_nlp = TransformersLanguage(trf_name=name)
    p_nlp.add_pipe(p_nlp.create_pipe("sentencizer"))
    p_nlp.add_pipe(TransformersWordPiecer.from_pretrained(p_nlp.vocab, trf_name=name))
    p_nlp.add_pipe(TransformersTok2Vec.from_pretrained(p_nlp.vocab, name=name))
    return p_nlp
