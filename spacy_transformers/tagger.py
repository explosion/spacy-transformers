from typing import List

import numpy
from thinc.api import chain, Model, xp2torch, to_numpy, with_array, Linear
from thinc.types import Floats2d

from spacy.util import registry

from .pipeline import TransformerListener
from .wrapper import TransformerModelByName
from .types import TransformerData, FullTransformerBatch
from .util import find_last_hidden


@registry.architectures.register("spacy.TransformerLinear.v1")
def transformer_linear_v1(nO: int=None, nI: int=None) -> Model[List[TransformerData], List[TransformerData]]:
    linear = Linear(nO=nO, nI=nI)
    return Model(
        "trf-linear",
        forward,
        layers=[with_array(linear)],
        init=init,
        dims={"nO": nO, "nI": nI},
        refs={"linear": linear}
    )


def init(model, X=None, Y=None): 
    X_ = _get_2d(X) if X is not None else None
    Y_ = _get_2d(Y) if Y is not None else None
    model.layers[0].initialize(X=X_, Y=Y_)
    model.set_dim("nO", model.get_ref("linear").get_dim("nO"))
    model.set_dim("nI", model.get_ref("linear").get_dim("nI"))


def forward(model, X: List[TransformerData], is_train: bool):
    print("x.shape", [x.tensors[-1].shape for x in X])
    X2d = _get_2d(X)
    Y2d, get_dX2d = model.layers[0](X2d, is_train)

    def backprop_trf_linear(dY: List[TransformerData]) -> List[TransformerData]:
        dX2d = get_dX2d(_get_2d(dY))
        print([dx.shape for dx in dX2d])
        dX = _replace_last_hidden(dX2d, dY)
        print([dx.tensors[-1].shape for dx in dX])
        return dX

    return _replace_last_hidden(Y2d, X), backprop_trf_linear


def _get_2d(trf_datas: List[TransformerData]) -> List[Floats2d]:
    outputs = []
    for trf in trf_datas:
        t_i = find_last_hidden(trf.tensors)
        outputs.append(trf.tensors[t_i].reshape((-1, trf.tensors[t_i].shape[-1])))
    return outputs


def _replace_last_hidden(last_hidden2d: List[Floats2d], trf_datas: List[TransformerData]) -> List[TransformerData]:
    assert len(last_hidden2d) == len(trf_datas)
    outputs = []
    for lh, trf in zip(last_hidden2d, trf_datas):
        t_i = find_last_hidden(trf.tensors)
        tensors = list(trf.tensors)
        shape = (trf.tensors[t_i].shape[0], trf.tensors[t_i].shape[1], lh.shape[1])
        tensors[t_i] = lh.reshape(shape)
        outputs.append(TransformerData(
            spans=trf.spans,
            tokens=trf.tokens,
            tensors=tensors,
            align=trf.align
        ))
    return outputs
