from spacy.lang.en import English
from spacy.training import Example
from spacy.util import load_config_from_str

CONFIG = """
[nlp]
lang = "en"
pipeline = ["transformer", "tagger"]

[components]

[components.transformer]
factory = "transformer"

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
"""


TRAIN_DATA = [
    ("I like green eggs", {"tags": ["N", "V", "J", "N"]}),
    ("Eat blue ham", {"tags": ["V", "J", "N"]}),
]


def test_empty_doc():
    """Test that an empty document gets processed correctly
    """
    nlp = English.from_config(load_config_from_str(CONFIG))
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    texts = ["first", "second", "third", "fourth", "and", "then", "some", ""]

    # run as normal
    nlp.select_pipes(enable=["transformer", "tagger"])
    docs1 = list(nlp.pipe(texts, batch_size=1))
    docs2 = list(nlp.pipe(texts, batch_size=4))
    assert [doc[0].tag_ for doc in docs1[:-1]] == [doc[0].tag_ for doc in docs2[:-1]]

    # disable the transformer (the listener will produce random output)
    nlp.select_pipes(enable=["tagger"])
    docs1 = list(nlp.pipe(texts, batch_size=1))
    docs2 = list(nlp.pipe(texts, batch_size=4))
    assert [doc[0].tag_ for doc in docs1[:-1]] == [doc[0].tag_ for doc in docs2[:-1]]
