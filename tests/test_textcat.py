import pytest
from spacy_transformers.pipeline import TransformersTextCategorizer
from spacy_transformers.util import PIPES
from spacy.gold import GoldParse


@pytest.fixture(
    params=["softmax_last_hidden", "softmax_pooler_output", "softmax_class_vector"]
)
def textcat(name, nlp, request):
    arch = request.param
    width = nlp.get_pipe(PIPES.tok2vec).model.nO
    textcat = TransformersTextCategorizer(nlp.vocab, token_vector_width=width)
    textcat.add_label("Hello")
    config = {"architecture": arch, "trf_name": name}
    if name.startswith("gpt2") and arch in (
        "softmax_pooler_output",
        "softmax_class_vector",
    ):
        with pytest.raises(ValueError):
            textcat.begin_training(**config)
        textcat.begin_training(trf_name=name, architecture="softmax_last_hidden")
    elif name.startswith("xlnet") and arch == "softmax_pooler_output":
        with pytest.raises(ValueError):
            textcat.begin_training(**config)
        textcat.begin_training(trf_name=name, architecture="softmax_last_hidden")
    else:
        textcat.begin_training(**config)
    return textcat


def test_textcat_init(nlp):
    textcat = TransformersTextCategorizer(nlp.vocab)
    assert textcat.labels == ()
    textcat.add_label("Hello")
    assert textcat.labels == ("Hello",)


def test_textcat_call(textcat, nlp):
    doc = nlp("hello world")
    for label in textcat.labels:
        assert label not in doc.cats
    doc = textcat(doc)
    for label in textcat.labels:
        assert label in doc.cats


def test_textcat_update(textcat, nlp):
    doc = nlp("hello world")
    optimizer = nlp.resume_training()
    cats = {"Hello": 1.0}
    losses = {}
    textcat.update([doc], [GoldParse(doc, cats=cats)], sgd=optimizer, losses=losses)
    assert PIPES.textcat in losses


def test_textcat_update_multi_sentence(textcat, nlp):
    doc = nlp("Hello world. This is sentence 2.")
    assert len(list(doc.sents)) == 2
    optimizer = nlp.resume_training()
    cats = {"Hello": 1.0}
    losses = {}
    textcat.update([doc], [GoldParse(doc, cats=cats)], sgd=optimizer, losses=losses)
    assert PIPES.textcat in losses


def test_textcat_update_batch(textcat, nlp):
    doc1 = nlp("Hello world. This is sentence 2.")
    doc2 = nlp("Hi again. This is sentence 4.")
    assert len(list(doc1.sents)) == 2
    assert len(list(doc2.sents)) == 2
    optimizer = nlp.resume_training()
    golds = [GoldParse(doc1, cats={"Hello": 1.0}), GoldParse(doc2, cats={"Hello": 0.0})]
    losses = {}
    textcat.update([doc1, doc2], golds, sgd=optimizer, losses=losses)
    assert PIPES.textcat in losses
