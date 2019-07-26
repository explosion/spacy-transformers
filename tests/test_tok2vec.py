import pytest

from spacy_pytorch_transformers import PyTT_Language, PyTT_TokenVectorEncoder
import pytorch_transformers as pytt

@pytest.fixture(scope="session")
def name():
    return "bert-base-uncased"

@pytest.fixture
def texts():
    return ["the cat sat on the mat.", "hello world."]

@pytest.fixture
def pytt_tokenizer(name):
    return pytt.BertTokenizer.from_pretrained(name)

@pytest.fixture
def nlp(pytt_tokenizer):
    nlp =  PyTT_Language()
    nlp.pytt_tokenizer = pytt_tokenizer
    return nlp

@pytest.fixture(scope="session")
def tok2vec(name):
    return PyTT_TokenVectorEncoder.from_pretrained(name, batch_by_length=True)

@pytest.fixture
def docs(nlp, texts):
    return [nlp.make_doc(text) for text in texts]

def test_from_pretrained(tok2vec, docs):
    docs_out = list(tok2vec.pipe(docs))
    assert len(docs_out) == len(docs)
    for doc in docs_out:
        assert doc.tensor.shape == (len(doc), tok2vec.model.nO)
        assert doc.tensor.sum() == doc._.pytt_outputs.last_hidden_state[1:-1].sum()

@pytest.mark.parametrize("text1,text2,is_similar,threshold", [
    ("The dog barked.", "The puppy barked.", True, 0.5),
    ("i ate an apple.", "an apple ate i.", False, 0.8)
    ("rats are cute", "cats please me", True, 0.6)
])
def test_similarity(nlp, tok2vec, text1, text2, is_similar, threshold):
    nlp.add_pipe(tok2vec)
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
    if is_similar:
        assert similarity >= threshold
    else:
        assert similarity < threshold


def test_sentence_batching():
