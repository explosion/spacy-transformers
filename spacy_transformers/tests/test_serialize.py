import pytest
import copy
import spacy
from spacy import Language
from spacy.lang.en import English
from spacy.tests.util import assert_docs_equal
from spacy.tokens import Doc
from spacy.util import make_tempdir
from spacy import util
import srsly
from thinc.api import Config, get_current_ops
from numpy.testing import assert_array_equal

from .. import TransformerData


DEFAULT_CONFIG = {
    "model": {
        "@architectures": "spacy-transformers.TransformerModel.v3",
        "name": "hf-internal-testing/tiny-random-DistilBertModel",
        "tokenizer_config": {"use_fast": False},
    }
}


def test_serialize_transformer_data():
    data = {"x": TransformerData.empty()}
    bytes_data = srsly.msgpack_dumps(data)
    new_data = srsly.msgpack_loads(bytes_data)
    assert isinstance(new_data["x"], TransformerData)

    nlp = Language()
    nlp.add_pipe(
        "transformer",
        config={
            "model": {
                "name": "hf-internal-testing/tiny-random-DistilBertModel",
                "transformer_config": {"output_attentions": True},
            }
        },
    )
    nlp.initialize()
    doc = nlp("This is a test.")
    b = doc.to_bytes()
    reloaded_doc = Doc(nlp.vocab)
    reloaded_doc.from_bytes(b)
    assert_docs_equal(doc, reloaded_doc)
    ops = get_current_ops()
    for key in doc._.trf_data.model_output:
        assert_array_equal(
            ops.to_numpy(ops.asarray(doc._.trf_data.model_output[key])),
            ops.to_numpy(ops.asarray(reloaded_doc._.trf_data.model_output[key])),
        )


def test_transformer_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer", config=DEFAULT_CONFIG)
    trf2.from_bytes(trf_bytes)


def test_initialized_transformer_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    nlp.initialize()
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer", config=DEFAULT_CONFIG)
    trf2.from_bytes(trf_bytes)

    assert trf2.model.tokenizer.is_fast is False


def test_initialized_transformer_todisk():
    nlp = Language()
    trf = nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    nlp.initialize()
    with make_tempdir() as d:
        trf.to_disk(d)
        nlp2 = Language()
        trf2 = nlp2.add_pipe("transformer", config=DEFAULT_CONFIG)
        trf2.from_disk(d)

        assert trf2.model.tokenizer.is_fast is False

    fast_config = copy.deepcopy(DEFAULT_CONFIG)
    fast_config["model"]["tokenizer_config"]["use_fast"] = True
    nlp = Language()
    trf = nlp.add_pipe("transformer", config=fast_config)
    nlp.initialize()
    with make_tempdir() as d:
        trf.to_disk(d)
        nlp2 = Language()
        trf2 = nlp2.add_pipe("transformer", config=fast_config)
        trf2.from_disk(d)

        assert trf2.model.tokenizer.is_fast is True


def test_transformer_pipeline_tobytes():
    nlp = Language()
    nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    nlp.initialize()
    assert nlp.pipe_names == ["transformer"]
    nlp_bytes = nlp.to_bytes()

    nlp2 = Language()
    nlp2.add_pipe("transformer", config=DEFAULT_CONFIG)
    nlp2.from_bytes(nlp_bytes)
    assert nlp2.pipe_names == ["transformer"]


def test_transformer_pipeline_todisk():
    nlp = English()
    nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    nlp.initialize()
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        assert nlp2.pipe_names == ["transformer"]


def test_transformer_pipeline_todisk_settings():
    nlp = English()
    trf = nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    nlp.initialize()
    # initially no attentions
    assert trf.model.tokenizer.model_max_length == 512
    assert trf.model.transformer.config.output_attentions is False
    assert "attentions" not in nlp("test")._.trf_data.model_output
    # modify model_max_length (note that modifications to
    # tokenizer.model_max_length for transformers<4.25 are not serialized by
    # save_pretrained, see: https://github.com/explosion/spaCy/discussions/7393)
    trf.model.tokenizer.init_kwargs["model_max_length"] = 499
    # transformer>=4.25, model_max_length is saved and init_kwargs changes are
    # clobbered, so do both for this test
    trf.model.tokenizer.model_max_length = 499
    # add attentions on-the-fly
    trf.model.transformer.config.output_attentions = True
    assert nlp("test")._.trf_data.model_output.attentions is not None
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        assert nlp2.pipe_names == ["transformer"]
        trf2 = nlp2.get_pipe("transformer")
        # model_max_length is preserved
        assert trf2.model.tokenizer.model_max_length == 499
        # output_attentions setting is preserved
        assert trf2.model.transformer.config.output_attentions is True
        assert nlp2("test")._.trf_data.model_output.attentions is not None
        # the init configs are empty SimpleFrozenDicts
        assert trf2.model._init_tokenizer_config == {}
        with pytest.raises(NotImplementedError):
            trf2.model._init_tokenizer_config["use_fast"] = False


def test_transformer_pipeline_todisk_before_initialize():
    nlp = English()
    nlp.add_pipe("transformer", config=DEFAULT_CONFIG)
    with make_tempdir() as d:
        # serialize before initialization
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        nlp2.initialize()
        assert "last_hidden_state" in nlp2("test")._.trf_data.model_output


inline_cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-transformers.Tok2VecTransformer.v3"
    name = "hf-internal-testing/tiny-random-DistilBertModel"
    tokenizer_config = {"use_fast": true}
    transformer_config = {"output_attentions": false}
    grad_factor = 1.0

    [components.tagger.model.tok2vec.get_spans]
    @span_getters = "spacy-transformers.strided_spans.v1"
    window = 256
    stride = 256

    [components.tagger.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"
    """


def test_inline_transformer_tobytes():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")
    tagger_bytes = tagger.to_bytes()

    nlp2 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    tagger2 = nlp2.get_pipe("tagger")
    tagger2.from_bytes(tagger_bytes)


def test_initialized_inline_transformer_tobytes():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["tagger"]
    tagger = nlp.get_pipe("tagger")
    tagger.add_label("V")
    nlp.initialize()
    tagger_bytes = tagger.to_bytes()

    nlp2 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    tagger2 = nlp2.get_pipe("tagger")
    tagger2.from_bytes(tagger_bytes)
    assert list(tagger2.labels) == ["V"]


def test_inline_transformer_todisk():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["tagger"]
    tagger = nlp.get_pipe("tagger")
    tagger.add_label("V")
    with make_tempdir() as d:
        tagger.to_disk(d)
        nlp2 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
        tagger2 = nlp2.get_pipe("tagger")
        tagger2.from_disk(d)
        assert list(tagger2.labels) == ["V"]


def test_initialized_inline_transformer_todisk():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["tagger"]
    tagger = nlp.get_pipe("tagger")
    tagger.add_label("V")
    nlp.initialize()
    with make_tempdir() as d:
        tagger.to_disk(d)
        nlp2 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
        tagger2 = nlp2.get_pipe("tagger")
        tagger2.from_disk(d)
        assert list(tagger2.labels) == ["V"]


def test_inline_transformer_pipeline_tobytes():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["tagger"]
    tagger = nlp.get_pipe("tagger")
    tagger.add_label("V")
    nlp.initialize()
    nlp_bytes = nlp.to_bytes()

    nlp2 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    nlp2.from_bytes(nlp_bytes)
    assert nlp2.pipe_names == ["tagger"]


def test_inline_transformer_pipeline_todisk():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["tagger"]
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        assert nlp2.pipe_names == ["tagger"]


def test_initialized_inline_transformer_pipeline_todisk():
    orig_config = Config().from_str(inline_cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["tagger"]
    tagger = nlp.get_pipe("tagger")
    tagger.add_label("V")
    nlp.initialize()
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        assert nlp2.pipe_names == ["tagger"]
        tagger2 = nlp2.get_pipe("tagger")
        assert list(tagger2.labels) == ["V"]
