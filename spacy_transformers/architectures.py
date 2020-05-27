from typing import List
from thinc.api import Model, chain
from thinc.types import Ragged, Floats2d
from spacy.tokens import Doc
from .layers import TransformerModel, TransformerListener
from .layers import trfs2arrays, split_trf_batch
from .util import registry


@registry.architectures.register("spacy-transformers.Tok2VecListener.v1")
def transformer_listener_tok2vec_v1(
    pooling: Model[Ragged, Floats2d], grad_factor: float = 1.0
) -> Model[List[Doc], List[Floats2d]]:
    return chain(TransformerListener("transformer"), trfs2arrays(pooling, grad_factor),)


@registry.architectures.register("spacy-transformers.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(
    name: str,
    get_spans,
    tokenizer_config,
    pooling: Model[Ragged, Floats2d],
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    return chain(
        TransformerModel(name, get_spans, tokenizer_config),
        split_trf_batch(),
        trfs2arrays(pooling, grad_factor),
    )


registry.architectures.register(
    "spacy-transformers.TransformerModel.v1", func=TransformerModel
)
