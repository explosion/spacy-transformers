from typing import List, Callable, Union
from thinc.api import Model, chain
from thinc.types import Ragged, Floats2d
from .layers import TransformerListener, trfs2arrays, split_trf_batch
from .data_classes import TransformerData
from .util import registry


@registry.architectures.register("spacy-transformers.listener.v1")
def transformer_listener_tok2vec_v1(
    pooling: Model[Ragged, Floats2d], width: int, grad_factor: float = 1.0
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        TransformerListener("transformer", width=width),
        trfs2arrays(pooling, width, grad_factor),
    )


@registry.architectures.register("spacy-transformers.Tok2Vec.v1")
def transformer_tok2vec_v1(
    transformer,
    pooling,
    get_spans,
    grad_factor: float = 1.0,
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        transformer,
        split_trf_batch(),
        trfs2arrays(pooling, width, grad_factor)
    )


@registry.architectures.register("spacy-transformers.empty_transformer.v1")
def empty_transformer_v1(get_spans: Callable, config) -> Model[List[Doc], TransformerData]:
    return Model(
        "transformer",
        forward,
        layers=[],
        attrs={
            "tokenizer": None,
            "get_spans": get_spans,
            "load": partial(load_transformer, get_spans=get_spans, config=config)
        },
        dims={"nO": None},
    )


@registry.architectures.register("spacy-transformers.load_transformer.v1")
def load_transformer_v1(
    load_from: Union[str, Path], get_spans: Callable, config: dict
) -> Model[List[Doc], TransformerData]:
    transformer = AutoModel.from_pretrained(load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from, **config)
    if isinstance(model.ops, CupyOps):
        transformer.cuda()
    return TransformerModel(transformer, tokenizer, get_spans)
