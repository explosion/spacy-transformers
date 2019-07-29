import pytest
from numpy.testing import assert_equal
from spacy_pytorch_transformers import PyTT_Language, PyTT_TokenVectorEncoder
from spacy_pytorch_transformers import PyTT_WordPiecer
from spacy.vocab import Vocab
import pickle

from .util import make_tempdir, is_valid_tensor


@pytest.fixture(scope="session")
def name():
    return "bert-base-uncased"


@pytest.fixture
def texts():
    return ["the cat sat on the mat.", "hello world."]


@pytest.fixture
def nlp(name):
    nlp = PyTT_Language(pytt_name=name)
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    wordpiecer = PyTT_WordPiecer.from_pretrained(nlp.vocab, pytt_name=name)
    nlp.add_pipe(wordpiecer)
    return nlp


@pytest.fixture(scope="session")
def tok2vec(name):
    cfg = {"batch_by_length": True}
    vocab = Vocab()
    return PyTT_TokenVectorEncoder.from_pretrained(vocab, name, **cfg)


@pytest.fixture
def docs(nlp, texts):
    return [nlp(text) for text in texts]


def test_from_pretrained(tok2vec, docs):
    docs_out = list(tok2vec.pipe(docs))
    assert len(docs_out) == len(docs)
    for doc in docs_out:
        diff = doc.tensor.sum() - doc._.pytt_last_hidden_state.sum()
        assert abs(diff) <= 1e-2


@pytest.mark.parametrize(
    "text1,text2,is_similar,threshold",
    [
        ("The dog barked.", "The puppy barked.", True, 0.5),
        ("i ate an apple.", "an apple ate i.", False, 0.8),
        ("rats are cute", "cats please me", True, 0.6),
    ],
)
def test_similarity(nlp, tok2vec, text1, text2, is_similar, threshold):
    nlp.add_pipe(tok2vec)
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
    if is_similar:
        assert similarity >= threshold
    else:
        assert similarity < threshold


def test_tok2vec_to_from_bytes(tok2vec, docs):
    doc = tok2vec(docs[0])
    assert is_valid_tensor(doc.tensor)
    bytes_data = tok2vec.to_bytes()
    new_tok2vec = PyTT_TokenVectorEncoder(Vocab(), **tok2vec.cfg)
    with pytest.raises(ValueError):
        new_doc = new_tok2vec(docs[0])
    new_tok2vec.from_bytes(bytes_data)
    new_doc = new_tok2vec(docs[0])
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)


def test_tok2vec_to_from_disk(tok2vec, docs):
    doc = tok2vec(docs[0])
    assert is_valid_tensor(doc.tensor)
    with make_tempdir() as tempdir:
        file_path = tempdir / "tok2vec"
        tok2vec.to_disk(file_path)
        new_tok2vec = PyTT_TokenVectorEncoder(Vocab())
        new_tok2vec.from_disk(file_path)
    new_doc = new_tok2vec(docs[0])
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)


@pytest.mark.skip
def test_tok2vec_pickle_dumps_loads(tok2vec, docs):
    doc = tok2vec(docs[0])
    assert is_valid_tensor(doc.tensor)
    pkl_data = pickle.dumps(tok2vec)
    new_tok2vec = pickle.loads(pkl_data)
    new_doc = new_tok2vec(docs[0])
    assert is_valid_tensor(new_doc.tensor)
    assert_equal(doc.tensor, new_doc.tensor)
