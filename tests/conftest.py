import pytest
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer
from spacy_pytorch_transformers import PyTT_TokenVectorEncoder

MODEL_NAMES = ["bert-base-uncased", "gpt2", "xlnet-base-cased"]


@pytest.fixture(scope="session", params=MODEL_NAMES)
def name(request):
    return request.param


@pytest.fixture(scope="session")
def nlp(name):
    p_nlp = PyTT_Language(pytt_name=name)
    p_nlp.add_pipe(p_nlp.create_pipe("sentencizer"))
    p_nlp.add_pipe(PyTT_WordPiecer.from_pretrained(p_nlp.vocab, pytt_name=name))
    p_nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(p_nlp.vocab, name=name))
    return p_nlp
