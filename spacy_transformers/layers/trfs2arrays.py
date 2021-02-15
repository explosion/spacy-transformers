from typing import List
from thinc.api import Model
from thinc.types import Ragged, Floats2d, FloatsXd
from ..data_classes import TransformerData
from ..util import find_last_hidden
from ..align import apply_alignment


def trfs2arrays(
    pooling: Model[Ragged, Floats2d], grad_factor: float
) -> Model[List[TransformerData], List[Floats2d]]:
    """Pool transformer data into token-aligned tensors."""
    return Model(
        "trfs2arrays", forward, layers=[pooling], attrs={"grad_factor": grad_factor}
    )


def forward(model: Model, trf_datas: List[TransformerData], is_train: bool):
    pooling: Model[Ragged, Floats2d] = model.layers[0]
    grad_factor = model.attrs["grad_factor"]
    outputs = []
    backprops = []
    for trf_data in trf_datas:
        if len(trf_data.tensors) > 0:
            t_i = find_last_hidden(trf_data.tensors)
            tensor_t_i = trf_data.tensors[t_i]
            if tensor_t_i.size == 0:
                # account for empty trf_data in the batch
                outputs.append(model.ops.alloc2f(0, 0))
            else:
                src = model.ops.reshape2f(tensor_t_i, -1, trf_data.width)
                dst, get_d_src = apply_alignment(model.ops, trf_data.align, src)
                output, get_d_dst = pooling(dst, is_train)
                outputs.append(output)
                backprops.append((get_d_dst, get_d_src))
        else:
            outputs.append(model.ops.alloc2f(0, 0))

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
                    wordpieces=trf_data.wordpieces,
                    align=trf_data.align
                )
            )
        return d_trf_datas

    assert len(outputs) == len(trf_datas)
    return outputs, backprop_trf_to_tensor
