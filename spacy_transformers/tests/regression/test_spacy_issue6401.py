import pytest
from spacy.training.example import Example
from spacy.util import make_tempdir
from spacy import util
from thinc.api import Config


TRAIN_DATA = [
    ("I'm so happy.", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("I'm so angry", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
]


cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["transformer","textcat"]

    [components]

    [components.textcat]
    factory = "textcat"

    [components.textcat.model]
    @architectures = "spacy.TextCatEnsemble.v2"

    [components.textcat.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0

    [components.textcat.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.transformer]
    factory = "transformer"
    """


def test_transformer_pipeline_textcat():
    """Test that a pipeline with just a transformer+textcat runs and trains properly.
    This used to throw an error because of shape inference issues -
    cf https://github.com/explosion/spaCy/issues/6401"""
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["transformer", "textcat"]
    train_examples = []

    for text, annotations in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    doc = nlp("We're interested at underwater basket weaving.")
    cats1 = doc.cats

    # ensure IO goes OK
    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)
        nlp2 = util.load_model_from_path(file_path)
        doc2 = nlp2("We're interested at underwater basket weaving.")
        cats2 = doc2.cats
        assert cats1 == cats2
