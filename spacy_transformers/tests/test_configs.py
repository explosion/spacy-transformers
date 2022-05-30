from functools import partial

import pytest
import spacy
from spacy.training import Example
from spacy.training.initialize import init_nlp
from spacy.util import CONFIG_SECTION_ORDER
from spacy.language import DEFAULT_CONFIG
from thinc.config import Config


TRAIN_TAGGER_DATA = [
    ("I like green eggs", {"tags": ["N", "V", "J", "N"]}),
    ("Eat blue ham", {"tags": ["V", "J", "N"]}),
]


cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["custom_transformer","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    upstream = "custom_transformer"

    [components.tagger.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"

    [components.custom_transformer]
    factory = "transformer"

    [corpora]
    @readers = toy_tagger_data.v1
    
    [initialize]
    
    [initialize.components]
    
    [initialize.components.tagger]
    labels = ["LABEL"]
    """


@pytest.mark.parametrize("config_string", [cfg_string])
def test_init_nlp(config_string):
    @spacy.registry.readers.register("toy_tagger_data.v1")
    def read_tagger_data():
        def parse_data(nlp, index):
            ex = TRAIN_TAGGER_DATA[index]
            yield Example.from_dict(nlp.make_doc(ex[0]), ex[1])

        return {
            "train": partial(parse_data, index=0),
            "dev": partial(parse_data, index=1),
        }

    config = spacy.util.load_config_from_str(config_string, interpolate=False)
    config = Config(DEFAULT_CONFIG, section_order=CONFIG_SECTION_ORDER).merge(config)
    nlp = init_nlp(config, use_gpu=False)
    assert nlp is not None

    tagger = nlp.get_pipe("tagger")
    transformer = nlp.get_pipe("custom_transformer")
    tagger_trf = tagger.model.get_ref("tok2vec").layers[0]
    assert tagger_trf.upstream_name == "custom_transformer"
    assert transformer.listeners[0] == tagger_trf
