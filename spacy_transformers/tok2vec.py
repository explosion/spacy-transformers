from typing import List

import numpy
from thinc.api import chain, Model, xp2torch, to_numpy
from thinc.types import Floats2d

from spacy.util import registry

from .pipeline import TransformerListener
from .wrapper import TransformerModelByName
from .types import TransformerData, FullTransformerBatch


@registry.architectures.register("spacy.Tok2VecTransformerListener.v1")
def transformer_listener_tok2vec_v1(width: int):
    tok2vec = chain(
        TransformerListener("transformer", width=width), trf_data_to_tensor(width)
    )
    tok2vec.set_dim("nO", width)
    return tok2vec


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(name: str, width: int):
    tok2vec = chain(
        TransformerModelByName(name),
        get_trf_data(),
        trf_data_to_tensor(width))
    tok2vec.set_dim("nO", width)
    return tok2vec


def get_trf_data() -> Model[FullTransformerBatch, List[TransformerData]]:
    def _forward(model, trf_full, is_train):
        def backprop(d_trf_datas):
            return trf_full.unsplit_by_doc([x.tensors for x in d_trf_datas])
        return trf_full.doc_data, backprop
    return Model("get-trf-data", _forward)


def trf_data_to_tensor(width) -> Model[List[TransformerData], List[Floats2d]]:
    return Model("trf-data-to-tensor", forward, dims={"nO": width})


def forward(model, trf_datas: List[TransformerData], is_train):
    outputs = []
    indices = []
    for trf_data in trf_datas:
        wp_array = to_numpy(trf_data.tensors[-1])
        shape = (len(trf_data.align), wp_array.shape[-1])
        outputs.append(numpy.zeros(shape, dtype="f"))
        indices.append([])
        for i, tok_align in enumerate(trf_data.align):
            wp_idx = tok_align[-1]
            outputs[-1][i] = wp_array[wp_idx]
            indices[-1].append(wp_idx)
    outputs = [model.ops.asarray(arr) for arr in outputs]

    def backprop(d_outputs: List[Floats2d]) -> List[TransformerData]:
        assert len(d_outputs) == len(trf_datas)
        d_trf_datas = []
        for trf_data, d_output, rows in zip(trf_datas, d_outputs, indices):
            d_tensors = [numpy.zeros((0, 0), dtype="f") for x in trf_data.tensors]
            d_tensors[-1] = numpy.zeros(trf_data.tensors[-1].shape, dtype="f")
            d_output = to_numpy(d_output)
            for i, wp_row in enumerate(rows):
                d_tensors[-1][wp_row] += d_output[i]
            d_trf_datas.append(
                TransformerData(
                    spans=trf_data.spans,
                    tokens=trf_data.tokens,
                    tensors=[model.ops.asarray(x) for x in d_tensors],
                    align=trf_data.align
                )
            )
        return trf_datas

    return outputs, backprop
