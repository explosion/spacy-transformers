from typing import List

import numpy
from thinc.api import chain, Model, xp2torch, to_numpy
from thinc.types import Floats2d

from spacy.util import registry

from .pipeline import TransformerListener
from .wrapper import TransformerModelByName
from .types import TransformerData, FullTransformerBatch


@registry.architectures.register("spacy.Tok2VecTransformerListener.v1")
def transformer_listener_tok2vec_v1(width: int, grad_factor: float=1.0):
    tok2vec = chain(
        TransformerListener("transformer", width=width),
        trf_data_to_tensor(width, grad_factor)
    )
    tok2vec.set_dim("nO", width)
    return tok2vec


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(name: str, width: int, grad_factor: float=1.0):
    tok2vec = chain(
        TransformerModelByName(name),
        get_trf_data(),
        trf_data_to_tensor(width, grad_factor))
    tok2vec.set_dim("nO", width)
    return tok2vec


def get_trf_data() -> Model[FullTransformerBatch, List[TransformerData]]:
    def _forward(model, trf_full, is_train):
        def backprop(d_trf_datas):
            return trf_full.unsplit_by_doc([x.tensors for x in d_trf_datas])
        return trf_full.doc_data, backprop
    return Model("get-trf-data", _forward)


def trf_data_to_tensor(width: int, grad_factor: float) -> Model[List[TransformerData], List[Floats2d]]:
    return Model("trf-data-to-tensor", forward, dims={"nO": width}, attrs={"grad_factor": grad_factor})


def forward(model, trf_datas: List[TransformerData], is_train):
    grad_factor = model.attrs["grad_factor"]
    outputs = []
    indices = []
    for trf_data in trf_datas:
        tensor_i = find_last_3d(trf_data.tensors)
        wp_array = to_numpy(trf_data.tensors[tensor_i])
        shape = (len(trf_data.align), wp_array.shape[-1])
        outputs.append(numpy.zeros(shape, dtype="f"))
        indices.append([])
        for i, tok_align in enumerate(trf_data.align):
            wp_idx0 = [x[0] for x in tok_align]
            wp_idx1 = [x[1] for x in tok_align]
            outputs[-1][i] = wp_array[wp_idx0, wp_idx1].mean(axis=0)
            indices[-1].append(tok_align)
    outputs = [model.ops.asarray(arr) for arr in outputs]

    def backprop(d_outputs: List[Floats2d]) -> List[TransformerData]:
        assert len(d_outputs) == len(trf_datas)
        d_trf_datas = []
        for trf_data, d_output, rows in zip(trf_datas, d_outputs, indices):
            t_i = find_last_3d(trf_data.tensors)
            d_tensors = [numpy.zeros(x.shape, dtype="f") for x in trf_data.tensors]
            d_tensors[tensor_i] = numpy.zeros(trf_data.tensors[t_i].shape, dtype="f")
            d_output = to_numpy(d_output)
            for i, entries in enumerate(rows):
                token_grad = d_output[i] / len(entries)
                for wp_row in entries:
                    d_tensors[t_i][wp_row] += token_grad
            d_tensors[t_i] *= grad_factor
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


def find_last_3d(tensors) -> int:
    for i, tensor in reversed(list(enumerate(tensors))):
        if len(tensor.shape) == 3:
            return i
    else:
        raise ValueError("No 3d tensors")
