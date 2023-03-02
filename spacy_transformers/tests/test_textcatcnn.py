import pytest
from packaging.version import Version

from spacy.training.example import Example
from spacy import util
import thinc
from thinc.api import Model, Config

# fmt: off
cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["textcat"]

    [components]

    [components.textcat]
    factory = "textcat"

    [components.textcat.model]
    @architectures = "spacy.TextCatCNN.v2"
    nO = null
    exclusive_classes = false

    [components.textcat.model.tok2vec]
    @architectures = "spacy-transformers.Tok2VecTransformer.v1"
    name = "roberta-base"
    tokenizer_config = {"use_fast": false}
    grad_factor = 1.0

    [components.textcat.model.tok2vec.get_spans]
    @span_getters = "spacy-transformers.strided_spans.v1"
    window = 256
    stride = 256

    [components.textcat.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"
    """
# fmt: on


# TODO: remove skip after requiring spacy>=3.5.1 or at the very latest, after
# dropping python 3.7 switch to importlib.metadata.version("thinc")
@pytest.mark.skipif(
    Version(thinc.__version__) < Version("8.1.8"), reason="Requires thinc>=8.1.8"
)
def test_textcatcnn():
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["textcat"]

    textcat = nlp.get_pipe("textcat")
    assert textcat.is_resizable is True

    train_examples = []
    doc = nlp.make_doc("ok")
    doc.cats["X"] = 1.0
    doc.cats["Y"] = 0.0
    train_examples.append(Example(doc, doc))

    nlp.initialize(lambda: train_examples)
