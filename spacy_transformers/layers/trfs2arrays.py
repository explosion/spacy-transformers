from typing import List
from thinc.api import Model
from thinc.types import Ragged, Floats2d, FloatsXd
from ..data_classes import TransformerData
from ..util import find_last_hidden
from ..align import apply_alignment


def trfs2arrays(
    pooling: Model[Ragged, Floats2d], width: int, grad_factor: float
) -> Model[List[TransformerData], List[Floats2d]]:
    return Model(
        "trfs2arrays",
        forward,
        layers=[pooling],
        dims={"nO": width},
        attrs={"grad_factor": grad_factor},
    )


def forward(model: Model, trf_datas: List[TransformerData], is_train: bool):
    pooling: Model[Ragged, Floats2d] = model.layers[0]
    grad_factor = model.attrs["grad_factor"]
    outputs = []
    backprops = []
    for trf_data in trf_datas:
        t_i = find_last_hidden(trf_data.tensors)
        src = model.ops.reshape2f(trf_data.tensors[t_i], -1, trf_data.width)
        dst, get_d_src = apply_alignment(model.ops, trf_data.align, src)
        output, get_d_dst = pooling(dst, is_train)
        assert output.shape[0] == trf_data.align.lengths.shape[0]
        if model.ops.xp.isnan(output.sum()):
            raise ValueError("nan in output")
        outputs.append(output)
        backprops.append((get_d_dst, get_d_src))

    def backprop_trf_to_tensor(d_outputs: List[Floats2d]) -> List[TransformerData]:
        d_trf_datas = []
        zipped = zip(trf_datas, d_outputs, backprops)
        for trf_data, d_output, (get_d_dst, get_d_src) in zipped:
            d_tensors: List[FloatsXd] = [
                model.ops.alloc(x.shape, dtype=x.dtype) for x in trf_data.tensors
            ]
            d_dst = get_d_dst(d_output)
            d_src = get_d_src(d_dst)
            d_src *= grad_factor
            t_i = find_last_hidden(trf_data.tensors)
            d_tensors[t_i] = d_src.reshape(trf_data.tensors[t_i].shape)
            d_trf_datas.append(
                TransformerData(
                    tensors=d_tensors,
                    tokens=trf_data.tokens,
                    align=trf_data.align,
                )
            )
        return d_trf_datas

    return outputs, backprop_trf_to_tensor
