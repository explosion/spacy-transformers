import spacy
from spacy import Language
from spacy.lang.en import English
from spacy.util import make_tempdir
from spacy import util
import srsly
from thinc.api import Config

from .. import TransformerData
from ..util import make_tempdir


def test_serialize_transformer_data():
    data = {"x": TransformerData.empty()}
    bytes_data = srsly.msgpack_dumps(data)
    new_data = srsly.msgpack_loads(bytes_data)
    assert isinstance(new_data["x"], TransformerData)


def test_transformer_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer")
    trf2.from_bytes(trf_bytes)


def test_initialized_transformer_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    nlp.initialize()
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer")
    trf2.from_bytes(trf_bytes)


def test_initialized_transformer_todisk():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    nlp.initialize()
    with make_tempdir() as d:
        trf.to_disk(d)
        nlp2 = Language()
        trf2 = nlp2.add_pipe("transformer")
        trf2.from_disk(d)


def test_transformer_pipeline_tobytes():
    nlp = Language()
    nlp.add_pipe("transformer")
    nlp.initialize()
    assert nlp.pipe_names == ["transformer"]
    nlp_bytes = nlp.to_bytes()

    nlp2 = Language()
    nlp2.add_pipe("transformer")
    nlp2.from_bytes(nlp_bytes)
    assert nlp2.pipe_names == ["transformer"]


def test_transformer_pipeline_todisk():
    nlp = English()
    nlp.add_pipe("transformer")
    nlp.initialize()
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
        assert nlp2.pipe_names == ["transformer"]


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
    @architectures = "spacy-transformers.Tok2VecTransformer.v1"
    name = "albert-base-v2"
    tokenizer_config = {"use_fast": true}
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
