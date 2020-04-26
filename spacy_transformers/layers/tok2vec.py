from typing import List

import torch
from thinc.api import chain, Model
from thinc.types import Floats2d

from ..pipeline import TransformerListener
from ..types import TransformerOutput


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
    docs = trf_data.docs
    outputs = [numpy.zeros((len(doc), width), dtype="f") for doc in docs]
    indices = []
    for i, doc in enumerate(docs):
        indices.append([])
        for j, token in enumerate(doc):
            wp_idx = token._.trf_alignment[-1]
            outputs[i][j] = wp_array[wp_idx]
            indices.append(wp_idx)
    outputs = [model.ops.asarray(arr) for arr in outputs]

    def backprop(d_outputs):
        d_tensors = [torch.zeros_like(t) for t in trf_data.tensors]
        for i in range(len(d_outputs)):
            d_tensors[indices[i]] = d_outputs[i]
        
        return TransformerOutput(
            spans=trf_data.spans,
            tokens=trf_data.tokens,
            tensors=d_tensors
        )

    return outputs, backprop
