from typing import List

import numpy
from thinc.api import chain, Model, xp2torch
from thinc.types import Floats2d

from spacy.util import registry

from .pipeline import TransformerListener
from .wrapper import TransformerModelByName
from .types import TransformerData


@registry.architectures.register("spacy.Tok2VecTransformerListener.v1")
def transformer_listener_tok2vec_v1(width: int):
    tok2vec = chain(
        TransformerListener("transformer", width=width), trf_data_to_tensor(width)
    )
    tok2vec.set_dim("nO", width)
    return tok2vec


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(name: str, width: int):
    tok2vec = chain(TransformerModelByName(name), trf_data_to_tensor(width))
    tok2vec.set_dim("nO", width)
    return tok2vec


def trf_data_to_tensor(width) -> Model[List[TransformerData], List[Floats2d]]:
    return Model("trf-data-to-tensor", forward, dims={"nO": width})


def forward(model, trf_datas: List[TransformerData], is_train):
    outputs = []
    indices = []
    for trf_data in trf_datas:
        wp_array = trf_data.tensors[-1]
        outputs.append(numpy.zeros(wp_array.shape, dtype="f"))
        indices.append([])
        for i, tok_align in enumerate(trf_data.align):
            wp_idx = tok_align[-1]
            outputs[-1][i] = wp_array[wp_idx]
            indices[-1].append(wp_idx)
    outputs = [model.ops.asarray(arr) for arr in outputs]

    def backprop(d_outputs):
        d_tensors = [numpy.zeros(t.shape, dtype="f") for t in d_outputs]
        for i in range(len(d_outputs)):
            d_output = xp2torch(d_outputs[i])
            for j, token_indices in enumerate(indices[i]):
                for entry in token_indices:
                    d_tensors[-1][i, j] += d_output[j]

        return TransformerData(
            spans=trf_data.spans,
            tokens=trf_data.tokens,
            tensors=d_tensors,
            ops=trf_data.ops,
        )

    return outputs, backprop
