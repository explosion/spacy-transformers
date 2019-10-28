import pytest
from spacy_transformers.pipeline import TransformersTextCategorizer
from spacy_transformers.util import PIPES
from spacy.gold import GoldParse


def textcat(name, nlp, request):
    arch = request.param
    tensor_size = nlp.get_pipe(PIPES.tok2vec).model.nO
    ner = TransformersEntityRecognizer(nlp.vocab)
    ner.add_label("PERSON")
    config = {"trf_name": name, "tensor_size": tensor_size}
    ner.begin_training(**config)
    return ner

@pytest.mark.xfail
def test_ner_init(nlp):
    ner = TransformersEntityRecognizer(nlp.vocab)
    assert textcat.labels == ()
    textcat.add_label("PERSON")
    assert textcat.labels == ("PERSON",)


@pytest.mark.xfail
def test_ner_call(ner, nlp):
    doc = nlp("hello world")


@pytest.mark.xfail
def test_ner_update(ner, nlp):
    doc = nlp("hello matt")
    optimizer = nlp.resume_training()
    ents = ["O", "U-PERSON"]
    losses = {}
    ner.update([doc], [GoldParse(doc, entities=ents)], sgd=optimizer, losses=losses)
    assert PIPES.ner in losses


@pytest.mark.xfail
def test_ner_update_multi_sentence(ner, nlp):
    doc = nlp("Hello matt. This is ian.")
    assert len(list(doc.sents)) == 2
    optimizer = nlp.resume_training()
    ents = ["O", "U-PERSON", "O", "O", "O", "U-PERSON", "O"]
    losses = {}
    ner.update([doc], [GoldParse(doc, entities=ents)], sgd=optimizer, losses=losses)
    assert PIPES.ner in losses


@pytest.mark.xfail
def test_ner_update_batch(ner, nlp):
    doc1 = nlp("Hello world. This is sentence 2.")
    doc2 = nlp("Hi again. This is sentence 4.")
    ents1 = ["O"] * len(doc1)
    ents2 = ["O"] * len(doc2)
    assert len(list(doc1.sents)) == 2
    assert len(list(doc2.sents)) == 2
    optimizer = nlp.resume_training()
    golds = [GoldParse(doc1, entities=ents1), GoldParse(doc2, entities=ents2)]
    losses = {}
    ner.update([doc1, doc2], golds, sgd=optimizer, losses=losses)
    assert PIPES.ner in losses
