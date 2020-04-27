from typing import List

import torch
import numpy
from thinc.api import chain, Model, xp2torch
from thinc.types import Floats2d

from spacy.util import registry

from .pipeline import TransformerListener
from .types import TransformerOutput


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(width):
    tok2vec = chain(
        TransformerListener("transformer", width=width),
        trf_data_to_tensor(width)
    )
    tok2vec.set_dim("nO", width)
    return tok2vec


def trf_data_to_tensor(width) -> Model[TransformerOutput, List[Floats2d]]:
    return Model("trf-data-to-tensor", forward, dims={"nO": width})

    
def forward(model, trf_data, is_train):
    width = trf_data.width
    docs = trf_data.docs
    outputs = [numpy.zeros((len(doc), width), dtype="f") for doc in docs]
    indices = []
    wp_array = trf_data.arrays[-1]
    for i, doc in enumerate(docs):
        indices.append([])
        for j, token in enumerate(doc):
            if token._.trf_alignment:
                wp_idx = token._.trf_alignment[-1]
                outputs[i][j] = wp_array[wp_idx]
                indices.append(wp_idx)
            else:
                indices.append([])
    outputs = [model.ops.asarray(arr) for arr in outputs]

    def backprop(d_outputs):
        d_tensors = [torch.zeros_like(t) for t in trf_data.tensors]
        for i in range(len(d_outputs)):
            d_output = xp2torch(d_outputs[i])
            for j, token_indices in enumerate(indices[i]):
                for entry in token_indices:
                    d_tensors[-1][i, j] += d_output[j]
        
        return TransformerOutput(
            spans=trf_data.spans,
            tokens=trf_data.tokens,
            tensors=d_tensors,
            ops=trf_data.ops
        )

    return outputs, backprop
