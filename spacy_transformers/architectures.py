from typing import List, Callable, Union
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from thinc.api import Model, chain, CupyOps
from thinc.types import Ragged, Floats2d
from spacy.tokens import Doc
from .layers import TransformerModel, TransformerListener
from .layers import trfs2arrays, split_trf_batch
from .data_classes import TransformerData
from .util import registry


@registry.architectures.register("spacy-transformers.Tok2VecListener.v1")
def transformer_listener_tok2vec_v1(
    pooling: Model[Ragged, Floats2d], width: int, grad_factor: float = 1.0
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        TransformerListener("transformer", width=width),
        trfs2arrays(pooling, width, grad_factor),
    )


@registry.architectures.register("spacy-transformers.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(
    transformer, pooling, width: int, grad_factor: float = 1.0
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        transformer,
        split_trf_batch(),
        trfs2arrays(pooling, width, grad_factor)
    )


registry.architectures.register(
    "spacy-transformers.TransformerModel.v1",
    func=TransformerModel
)
