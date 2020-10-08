import pytest
from spacy.language import Language
from spacy.training.example import Example
from spacy.util import make_tempdir
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy import util
from spacy_transformers.layers import TransformerListener
from thinc.api import Model, Config
from numpy.testing import assert_equal

from .util import DummyTransformer
from ..pipeline_component import Transformer
from ..data_classes import TransformerData, FullTransformerBatch


@pytest.fixture
def vocab():
    return Vocab()


@pytest.fixture
def docs(vocab):
    return [
        Doc(vocab, words=["hello", "world"]),
        Doc(vocab, words=["this", "is", "another"]),
    ]


@pytest.fixture
def component(vocab):
    return Transformer(Vocab(), DummyTransformer())


def test_init(component):
    assert isinstance(component.vocab, Vocab)
    assert isinstance(component.model, Model)
    assert hasattr(component.set_extra_annotations, "__call__")
    assert component.listeners == []
    assert component.cfg == {"max_batch_items": 4096}


def test_predict(component, docs):
    trf_data = component.predict(docs)
    assert isinstance(trf_data, FullTransformerBatch)
    assert len(trf_data.tensors) == component.model.layers[0].attrs["depth"]
    n_tokens = trf_data.tokens["input_ids"].shape[1]
    width = component.model.layers[0].attrs["width"]
    assert trf_data.tensors[-1].shape == (len(docs), n_tokens, width)


def test_set_annotations(component, docs):
    trf_data = component.predict(docs)
    component.set_annotations(docs, trf_data)
    for doc in docs:
        assert isinstance(doc._.trf_data, TransformerData)


def test_set_extra_annotations(component, docs):
    Doc.set_extension("custom_attr", default="")

    def custom_annotation_setter(docs, trf_data):
        doc_data = list(trf_data.doc_data)
        for doc, data in zip(docs, doc_data):
            doc._.custom_attr = data

    component.set_extra_annotations = custom_annotation_setter
    trf_data = component.predict(docs)
    component.set_annotations(docs, trf_data)
    for doc in docs:
        assert isinstance(doc._.custom_attr, TransformerData)


def test_listeners(component, docs):
    docs = list(component.pipe(docs))
    for listener in component.listeners:
        assert listener.verify_inputs(docs)


TRAIN_DATA = [
    ("I like green eggs", {"tags": ["N", "V", "J", "N"]}),
    ("Eat blue ham", {"tags": ["V", "J", "N"]}),
]


def test_transformer_pipeline_simple():
    """Test that a simple pipeline with just a transformer at least runs"""
    nlp = Language()
    nlp.add_pipe("transformer")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.initialize()
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    doc = nlp("We're interested at underwater basket weaving.")
    assert doc


cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"
    
    [components.tagger.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    
    [components.tagger.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.transformer]
    factory = "transformer"
    """


def test_transformer_pipeline_tagger():
    """Test that a pipeline with just a transformer+tagger runs and trains properly"""
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["transformer", "tagger"]
    tagger = nlp.get_pipe("tagger")
    transformer = nlp.get_pipe("transformer")
    tagger_trf = tagger.model.get_ref("tok2vec").layers[0]
    assert isinstance(transformer, Transformer)
    assert isinstance(tagger_trf, TransformerListener)
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    # Check that the Transformer component finds it listeners
    assert transformer.listeners == []
    optimizer = nlp.initialize(lambda: train_examples)
    assert tagger_trf in transformer.listeners

    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    doc = nlp("We're interested at underwater basket weaving.")
    doc_tensor = tagger_trf.predict([doc])
    assert_equal(doc._.trf_data.tensors, doc_tensor[0].tensors)

    # ensure IO goes OK
    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)
        nlp2 = util.load_model_from_path(file_path)
        doc = nlp2("We're interested at underwater basket weaving.")
        tagger2 = nlp2.get_pipe("tagger")
        tagger_trf2 = tagger2.model.get_ref("tok2vec").layers[0]
        doc_tensor2 = tagger_trf2.predict([doc])
        assert_equal(doc_tensor2[0].tensors, doc_tensor[0].tensors)
