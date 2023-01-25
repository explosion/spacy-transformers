from typing import List, cast
from spacy.util import all_equal
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from thinc.api import Model
from thinc.types import Ragged, Floats2d
from ..data_classes import TransformerData
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

    # Cache the width of the model outputs from a non-empty output, if possible.
    width = 0
    for trf_data in trf_datas:
        if "last_hidden_state" in trf_data.model_output:
            last_hidden_state = trf_data.model_output.last_hidden_state
            assert len(last_hidden_state.shape) == 3  # [batch, seq_len, width]
            width = last_hidden_state.shape[2]
            break

    for trf_data in trf_datas:
        if "last_hidden_state" in trf_data.model_output:
            tensor_t_i = cast(BaseModelOutput, trf_data.model_output).last_hidden_state
            if tensor_t_i.size == 0:
                # This can happen during prediction/initialization if the transformer pipe was disabled/not executed and one of the inputs
                # was of length zero. This causes the listenener to generate a zero-sized (in the sequence length dim) TransformerData
                # output and pass it downstream.
                #
                # We also don't have to ensure that the backprops list stays in sync with the outputs as zero-length documents
                # are filtered out very early in the corpus generation stage of the training loop.
                assert not is_train
                outputs.append(model.ops.alloc2f(0, width))
            else:
                # This is the general case for non-zero length documents.
                src = model.ops.reshape2f(tensor_t_i, -1, trf_data.width)  # type: ignore
                dst, get_d_src = apply_alignment(model.ops, trf_data.align, src)
                output, get_d_dst = pooling(dst, is_train)
                outputs.append(output)
                backprops.append((get_d_dst, get_d_src))
        else:
            # This can happen during prediciton for zero-length documents. Since zero-length docs
            # are implicitly ignored in the span generation stage, the transformer model does not return any
            # predictions for them and subsequently, FullTransformerBatch.split_by_doc() generates an empty
            # TransformerData.
            assert not is_train
            outputs.append(model.ops.alloc2f(0, width))

    def backprop_trf_to_tensor(d_outputs: List[Floats2d]) -> List[TransformerData]:
        d_trf_datas = []
        to_zip = (trf_datas, d_outputs, backprops)
        assert all_equal(len(x) for x in to_zip)
        zipped = zip(*to_zip)
        for trf_data, d_output, (get_d_dst, get_d_src) in zipped:
            d_model_output = ModelOutput(
                last_hidden_state=model.ops.alloc(
                    trf_data.model_output.last_hidden_state.shape,  # type: ignore
                    dtype=trf_data.model_output.last_hidden_state.dtype,  # type: ignore
                )
            )
            d_dst = get_d_dst(d_output)
            d_src = get_d_src(d_dst)
            d_src *= grad_factor
            d_model_output["last_hidden_state"] = d_src.reshape(
                cast(BaseModelOutput, trf_data.model_output).last_hidden_state.shape
            )
            d_trf_datas.append(
                TransformerData(
                    model_output=d_model_output,
                    wordpieces=trf_data.wordpieces,
                    align=trf_data.align,
                )
            )
        return d_trf_datas

    assert len(outputs) == len(trf_datas)
    return outputs, backprop_trf_to_tensor
