import pytest
from packaging.version import Version
import torch
import spacy
from spacy.language import Language
from spacy.training.example import Example
from spacy.util import make_tempdir
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy import util
from thinc.api import Model, Config, get_current_ops, NumpyOps
from spacy.tests.util import assert_docs_equal

from .util import DummyTransformer, _assert_equal_tensors
from .. import TransformerModel
from ..pipeline_component import Transformer
from ..layers import TransformerListener
from ..data_classes import TransformerData, FullTransformerBatch


torch.set_num_threads(1)


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


@pytest.fixture(scope="module")
def simple_nlp():
    nlp = Language()
    nlp.add_pipe("transformer")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.initialize()
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    return nlp


def test_init(component):
    assert isinstance(component.vocab, Vocab)
    assert isinstance(component.model, Model)
    assert hasattr(component.set_extra_annotations, "__call__")
    assert component.listeners == []
    assert component.cfg == {"max_batch_items": 4096}


def test_predict(component, docs):
    trf_data = component.predict(docs)
    n_tokens = trf_data.wordpieces.input_ids.shape[1]
    width = component.model.layers[0].attrs["width"]
    assert isinstance(trf_data, FullTransformerBatch)
    assert (
        len(trf_data.model_output.last_hidden_state)
        == component.model.layers[0].attrs["depth"]
    )
    assert trf_data.model_output.last_hidden_state[0].shape == (
        len(docs),
        n_tokens,
        width,
    )


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
    (
        "I like green eggs",
        {"tags": ["N", "V", "J", "N"], "sent_starts": [True, False, True, False]},
    ),
    ("Eat blue ham", {"tags": ["V", "J", "N"], "sent_starts": [True, False, False]}),
]


def test_transformer_pipeline_simple(simple_nlp):
    """Test that a simple pipeline with just a transformer at least runs"""
    doc = simple_nlp("We're interested at underwater basket weaving.")
    assert doc


def test_transformer_pipeline_long_token(simple_nlp):
    """Test that a simple pipeline does not raise an error on texts that exceeds
    the model max length. We should truncate instead.
    """
    doc = simple_nlp("https://example.com/" + "a/" * 1000)
    assert len(doc) == 1


cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger","senter"]

    [components]

    [components.senter]
    factory = "senter"

    [components.senter.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.senter.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    upstream = "transformer"

    [components.senter.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    upstream = "transformer"

    [components.tagger.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.transformer]
    factory = "transformer"

    [components.transformer.model]
    @architectures = "spacy-transformers.TransformerModel.v3"
    name = "albert-base-v2"

    [components.transformer.model.transformer_config]
    output_attentions = true
    """


def test_transformer_pipeline_tagger_senter_listener():
    """Test that a pipeline with just a transformer+tagger+senter runs and
    trains properly"""
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["transformer", "tagger", "senter"]
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
    optimizer = nlp.initialize(lambda: train_examples)
    assert tagger_trf in transformer.listeners

    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    text = "We're interested at underwater basket weaving."
    doc = nlp(text)
    doc_tensor = tagger_trf.predict([doc])
    _assert_equal_tensors(doc._.trf_data.tensors, doc_tensor[0].tensors)

    # ensure IO goes OK
    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)
        nlp2 = util.load_model_from_path(file_path)
        doc2 = nlp2(text)
        tagger2 = nlp2.get_pipe("tagger")
        tagger_trf2 = tagger2.model.get_ref("tok2vec").layers[0]
        doc_tensor2 = tagger_trf2.predict([doc2])
        _assert_equal_tensors(doc_tensor2[0].tensors, doc_tensor[0].tensors)

        # make sure that this can be saved to directory once more
        file_path_2 = d / "trained_nlp_2"
        nlp2.to_disk(file_path_2)

    # ensure to_bytes / from_bytes works
    nlp_bytes = nlp.to_bytes()
    nlp3 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    nlp3.from_bytes(nlp_bytes)
    doc3 = nlp3(text)
    tagger3 = nlp3.get_pipe("tagger")
    tagger_trf3 = tagger3.model.get_ref("tok2vec").layers[0]
    doc_tensor3 = tagger_trf3.predict([doc3])
    _assert_equal_tensors(doc_tensor3[0].tensors, doc_tensor[0].tensors)


def test_transformer_sentencepiece_IO():
    """Test that a transformer using sentencepiece trains + IO goes OK"""
    orig_config = Config().from_str(cfg_string)
    orig_config["components"]["transformer"]["model"]["name"] = "camembert-base"
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")
    tagger_trf = tagger.model.get_ref("tok2vec").layers[0]
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    optimizer = nlp.initialize(lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    text = "We're interested at underwater basket weaving."
    doc = nlp(text)
    doc_tensor = tagger_trf.predict([doc])

    # ensure IO goes OK
    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)
        nlp2 = util.load_model_from_path(file_path)
        doc2 = nlp2(text)
        tagger2 = nlp2.get_pipe("tagger")
        tagger_trf2 = tagger2.model.get_ref("tok2vec").layers[0]
        doc_tensor2 = tagger_trf2.predict([doc2])
        _assert_equal_tensors(doc_tensor2[0].tensors, doc_tensor[0].tensors)

        # make sure that this can be saved to directory once more
        file_path_2 = d / "trained_nlp_2"
        nlp2.to_disk(file_path_2)

    # ensure to_bytes / from_bytes works
    nlp_bytes = nlp.to_bytes()
    nlp3 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    nlp3.from_bytes(nlp_bytes)
    doc3 = nlp3(text)
    tagger3 = nlp3.get_pipe("tagger")
    tagger_trf3 = tagger3.model.get_ref("tok2vec").layers[0]
    doc_tensor3 = tagger_trf3.predict([doc3])
    _assert_equal_tensors(doc_tensor3[0].tensors, doc_tensor[0].tensors)


def test_transformer_pipeline_empty():
    """Test that the pipeline doesn't fail with empty input"""
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    # train on empty doc
    optimizer = nlp.initialize()
    losses = {}
    empty_train_example = Example.from_dict(nlp.make_doc(""), {})
    nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.update([empty_train_example], sgd=optimizer, losses=losses)
    train_examples.append(empty_train_example)
    nlp.update(train_examples, sgd=optimizer, losses=losses)
    # Interleave an empty doc between non-empty ones
    train_examples.insert(1, Example.from_dict(nlp.make_doc(""), {}))
    nlp.update(train_examples, sgd=optimizer, losses=losses)

    # predict empty doc
    doc = nlp("")
    _assert_empty(doc._.trf_data)
    docs = nlp.pipe(["", ""])
    for doc in docs:
        _assert_empty(doc._.trf_data)
    nlp.pipe([])

    # predict combination of empty and non-empty
    doc = nlp("This is a sentence")
    normal_tags = [t.tag_ for t in doc]

    docs = list(nlp.pipe(["", "This is a sentence", "", ""]))
    _assert_empty(docs[0]._.trf_data)
    assert [t.tag_ for t in docs[0]] == []
    assert [t.tag_ for t in docs[1]] == normal_tags
    _assert_empty(docs[2]._.trf_data)
    _assert_empty(docs[3]._.trf_data)


def _assert_empty(trf_data):
    assert trf_data.wordpieces.strings == []
    assert trf_data.wordpieces.input_ids.size == 0
    assert trf_data.wordpieces.attention_mask.size == 0
    assert trf_data.tensors == ()
    assert len(trf_data.align.data) == 0


def test_replace_listeners():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    text = "This is awesome"
    examples = [Example.from_dict(nlp.make_doc(text), {"tags": ["A", "B", "C"]})]
    optimizer = nlp.initialize(lambda: examples)
    # verify correct configuration with transformer listener
    transformer = nlp.get_pipe("transformer")
    tagger = nlp.get_pipe("tagger")
    tagger_tok2vec = tagger.model.get_ref("tok2vec")
    tagger_listener = tagger_tok2vec.get_ref("listener")
    assert isinstance(tagger_listener, TransformerListener)
    assert transformer.listener_map["tagger"][0] == tagger_listener
    assert isinstance(transformer.model, TransformerModel)
    assert (
        nlp.config["components"]["transformer"]["model"]["@architectures"]
        == "spacy-transformers.TransformerModel.v3"
    )
    assert (
        nlp.config["components"]["tagger"]["model"]["tok2vec"]["@architectures"]
        == "spacy-transformers.TransformerListener.v1"
    )
    # train pipe before replacing listeners
    for i in range(2):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        doc = nlp(text)

    preds = [t.tag_ for t in doc]
    doc_tensor = tagger_tok2vec.predict([doc])

    # replace listener and verify predictions are still the same
    nlp.replace_listeners("transformer", "tagger", ["model.tok2vec"])
    tagger = nlp.get_pipe("tagger")
    tagger_tok2vec = tagger.model.get_ref("tok2vec")
    assert isinstance(tagger_tok2vec, Model)
    assert tagger_tok2vec.layers[0].layers[0].name == "transformer"
    assert (
        nlp.config["components"]["tagger"]["model"]["tok2vec"]["@architectures"]
        == "spacy-transformers.Tok2VecTransformer.v3"
    )
    doc2 = nlp(text)
    assert preds == [t.tag_ for t in doc2]
    pred_tensor = tagger_tok2vec.predict([doc2])
    _assert_equal_tensors(doc_tensor, pred_tensor)

    # attempt training with the new pipeline
    optimizer = nlp.resume_training()
    for i in range(2):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        assert losses["tagger"] > 0.0

    # check for presence of additional fields in model_output
    assert doc2._.trf_data.model_output.pooler_output is not None
    assert doc2._.trf_data.model_output.attentions is not None

    # ensure IO goes OK
    doc_tensor_trained = tagger_tok2vec.predict([doc])
    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)
        nlp2 = util.load_model_from_path(file_path)
        doc3 = nlp2(text)
        tagger2 = nlp2.get_pipe("tagger")
        tagger_tok2vec2 = tagger2.model.get_ref("tok2vec")
        pred_tensor = tagger_tok2vec2.predict([doc3])
        _assert_equal_tensors(doc_tensor_trained, pred_tensor)


def test_replace_listeners_invalid():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    text = "This is awesome"
    examples = [Example.from_dict(nlp.make_doc(text), {"tags": ["A", "B", "C"]})]
    optimizer = nlp.initialize(lambda: examples)
    for i in range(2):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
    with pytest.raises(ValueError):
        nlp.replace_listeners("invalid", "tagger", ["model.tok2vec"])
    with pytest.raises(ValueError):
        nlp.replace_listeners("transformer", "parser", ["model.tok2vec"])
    with pytest.raises(ValueError):
        nlp.replace_listeners("transformer", "tagger", ["model.yolo"])
    with pytest.raises(ValueError):
        nlp.replace_listeners("transformer", "tagger", ["model.tok2vec", "model.yolo"])


@pytest.fixture
def texts():
    data = [
        "Hello world.",
        "This is spacy.",
        "You can use multiprocessing with pipe method.",
        "Please try!",
    ]
    return data


def test_multiprocessing(simple_nlp, texts):
    ops = get_current_ops()
    if isinstance(ops, NumpyOps):
        texts = texts * 3
        expecteds = [simple_nlp(text) for text in texts]
        docs = simple_nlp.pipe(texts, n_process=2, batch_size=2)

        for doc, expected_doc in zip(docs, expecteds):
            assert_docs_equal(doc, expected_doc)


def test_frozen_listener():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    text = "This is awesome"
    examples = [Example.from_dict(nlp.make_doc(text), {"tags": ["A", "B", "C"]})]
    optimizer = nlp.initialize(lambda: examples)
    # train pipe before freezing listener
    for i in range(2):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        doc = nlp(text)

    transformer_bytes = nlp.get_pipe("transformer").to_bytes()
    tagger_bytes = nlp.get_pipe("tagger").to_bytes()

    # train further with frozen listener
    for i in range(2):
        losses = {}
        nlp.update(
            examples,
            sgd=optimizer,
            losses=losses,
            exclude=["transformer"],
            annotates=["transformer"],
        )
        doc = nlp(text)

    # only tagger was updated
    assert nlp.get_pipe("transformer").to_bytes() == transformer_bytes
    assert nlp.get_pipe("tagger").to_bytes() != tagger_bytes


def test_no_update_listener_in_predict():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    listener = nlp.get_pipe("tagger").model.get_ref("tok2vec").get_ref("listener")
    transformer = nlp.get_pipe("transformer")

    text = "This is awesome"
    examples = [Example.from_dict(nlp.make_doc(text), {"tags": ["A", "B", "C"]})]
    docs = [eg.predicted for eg in examples]
    nlp.initialize(lambda: examples)

    transformer.update(examples)
    assert listener._backprop is not None

    transformer.predict(docs)
    assert listener._backprop is not None


@pytest.mark.skipif(
    Version(spacy.__version__) < Version("3.5.4"), reason="Bug fixed in spaCy v3.5.4"
)
def test_source_replace_listeners():
    """Test that a pipeline with a transformer+tagger+senter and some replaced
    listeners runs and trains properly"""
    orig_config = """
    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger","senter"]

    [components]

    [components.senter]
    factory = "senter"

    [components.senter.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.senter.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    upstream = "transformer"

    [components.senter.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    upstream = "transformer"

    [components.tagger.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.transformer]
    factory = "transformer"

    [components.transformer.model]
    @architectures = "spacy-transformers.TransformerModel.v3"
    name = "distilbert-base-uncased"
    """
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["transformer", "tagger", "senter"]
    tagger = nlp.get_pipe("tagger")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)
    optimizer = nlp.initialize(lambda: train_examples)
    assert nlp.get_pipe("transformer").listening_components == ["tagger", "senter"]
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    with make_tempdir() as dir_path:
        nlp.to_disk(dir_path)
        base_model = str(dir_path)
        new_config = {
            "nlp": {
                "lang": "en",
                "pipeline": ["transformer", "tagger", "senter", "ner"],
            },
            "components": {
                "transformer": {"source": base_model},
                "tagger": {
                    "source": base_model,
                    "replace_listeners": ["model.tok2vec"],
                },
                "senter": {
                    "source": base_model,
                    "replace_listeners": ["model.tok2vec"],
                },
                "ner": {
                    "factory": "ner",
                    "model": {
                        "@architectures": "spacy.TransitionBasedParser.v2",
                        "state_type": "ner",
                        "tok2vec": {
                            "@architectures": "spacy-transformers.TransformerListener.v1",
                            "grad_factor": 1.0,
                            "upstream": "transformer",
                            "pooling": {"@layers": "reduce_mean.v1"},
                        },
                    },
                },
            },
        }
        new_nlp = util.load_model_from_config(new_config, auto_fill=True)
        for component in ("tagger", "senter"):
            assert (
                new_nlp.config["components"][component]["model"]["tok2vec"][
                    "@architectures"
                ]
                == "spacy-transformers.Tok2VecTransformer.v3"
            )
        assert new_nlp.get_pipe("transformer").listening_components == ["ner"]

        with make_tempdir() as new_dir_path:
            new_nlp.to_disk(new_dir_path)
            new_nlp_re = spacy.load(new_dir_path)
            for component in ("tagger", "senter"):
                assert (
                    new_nlp.config["components"][component]["model"]["tok2vec"][
                        "@architectures"
                    ]
                    == "spacy-transformers.Tok2VecTransformer.v3"
                )
            assert new_nlp_re.get_pipe("transformer").listening_components == ["ner"]
