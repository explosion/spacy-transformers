from typing import List

import numpy
from thinc.api import chain, Model, xp2torch, to_numpy
from thinc.types import Floats2d

from spacy.util import registry

from .pipeline import TransformerListener
from .wrapper import TransformerModelByName
from .tagger import transformer_linear_v1
from .util import find_last_hidden, TransformerData, FullTransformerBatch


@registry.architectures.register("spacy.Tok2VecTransformerListener.v1")
def transformer_listener_tok2vec_v1(width: int, grad_factor: float=1.0):
    tok2vec = chain(
        TransformerListener("transformer", width=width),
        trf_data_to_tensor(width, grad_factor)
    )
    return tok2vec


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(get_spans, name: str, width: int, grad_factor: float=1.0):
    tok2vec = chain(
        TransformerModelByName(name, get_spans=get_spans),
        get_trf_data(),
        trf_data_to_tensor(width, grad_factor))
    return tok2vec


def get_trf_data() -> Model[FullTransformerBatch, List[TransformerData]]:
    def _forward(model, trf_full, is_train):
        def backprop(d_trf_datas):
            return trf_full.unsplit_by_doc([x.tensors for x in d_trf_datas])
        return trf_full.doc_data, backprop
    return Model("get-trf-data", _forward)


def trf_data_to_tensor(width: int, grad_factor: float) -> Model[List[TransformerData], List[Floats2d]]:
    return Model("trf-data-to-tensor", forward, dims={"nO": width}, attrs={"grad_factor": grad_factor})


def forward(model: Model, trf_datas: List[TransformerData], is_train: bool):
    grad_factor = model.attrs["grad_factor"]
    outputs = []
    for trf_data in trf_datas:
        src = trf_data.tensors[find_last_hidden(trf_data.tensors)]
        dst = trf_data.get_tok_aligned(src)
        outputs.append(dst)

    def backprop(d_outputs: List[Floats2d]) -> List[TransformerData]:
        # TODO:
        # * Implement BatchAlignment.slice
        # * Test
        assert len(d_outputs) == len(trf_datas)
        d_trf_datas = []
        for trf_data, d_dst in zip(trf_datas, d_outputs):
            d_tensors = [model.ops.alloc(x.shape, dtype="f") for x in trf_data.tensors]
            d_src = d_tensors[find_last_hidden(d_tensors)]
            d_src = trf_data.get_wp_aligned(d_dst)
            d_src *= grad_factor
            d_trf_datas.append(
                TransformerData(
                    tensors=d_tensors,
                    spans=trf_data.spans,
                    tokens=trf_data.tokens,
                    wp2tok=trf_data.wp2tok,
                    tok2wp=trf_data.tok2wp,
                )
            )
        return d_trf_datas

    return outputs, backprop
