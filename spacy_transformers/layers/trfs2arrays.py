from typing import Callable, List, Optional, Tuple, cast
import numpy
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
        "trfs2arrays",
        forward,
        layers=[pooling],
        attrs={"grad_factor": grad_factor},
    )


def forward(model: Model, trf_datas: List[TransformerData], is_train: bool):
    pooling: Model[Ragged, Floats2d] = model.layers[0]
    grad_factor = model.attrs["grad_factor"]
    zero_outputs: List[Tuple[int, Floats2d]] = []
    backprops_alignment: List[Optional[Callable]] = []
    aligned_outputs: List[Tuple[int, Ragged]] = []

    # For zero-length documents, we could cache the output width by iterating
    # through the batch outputs and retrieving the shape of a non-zero length
    # Doc. This, however, is not fool-proof as one can pass an entire batch of
    # zero-length Docs to the transformer model (at least during prediction).
    # Instead of being conditionally correct, we'll explicitly leave the width as
    # zero in these cases as the effective length of the resultant tensor is zero anyway.
    output_width = 0

    for i, trf_data in enumerate(trf_datas):
        if "last_hidden_state" in trf_data.model_output:
            tensor_t_i = cast(BaseModelOutput, trf_data.model_output).last_hidden_state
            if tensor_t_i.size == 0:
                # This can happen during prediction/initialization if the transformer pipe was disabled/not executed and one of the inputs
                # was of length zero. This causes the listenener to generate a zero-sized (in the sequence length dim) TransformerData
                # output and pass it downstream.
                zero_outputs.append((i, model.ops.alloc2f(0, output_width)))
                backprops_alignment.append(None)
            else:
                # This is the general case for non-zero length documents.
                src = model.ops.reshape2f(tensor_t_i, -1, trf_data.width)  # type: ignore
                dst, get_d_src = apply_alignment(model.ops, trf_data.align, src)
                aligned_outputs.append((i, dst))
                backprops_alignment.append(get_d_src)
        else:
            # This can happen during prediction/training for zero-length documents. Since zero-length docs
            # are implicitly ignored in the span generation stage, the transformer model does not return any
            # predictions for them and subsequently, FullTransformerBatch.split_by_doc() generates an empty
            # TransformerData.
            zero_outputs.append((i, model.ops.alloc2f(0, output_width)))
            backprops_alignment.append(None)

    pooling_outputs, backprop_pooling = concat_pooling_forward(
        pooling, [dst for _, dst in aligned_outputs], is_train
    )

    # Interleave the zero and non-zero outputs into the final result.
    outputs: List[Optional[Floats2d]] = [None] * (
        len(zero_outputs) + len(aligned_outputs)
    )
    for i, zero_output in zero_outputs:
        outputs[i] = zero_output
    for (i, _), pooling_output in zip(aligned_outputs, pooling_outputs):
        outputs[i] = pooling_output

    def backprop_trf_to_tensor(d_outputs: List[Floats2d]) -> List[TransformerData]:
        d_trf_datas: List[TransformerData] = []

        # Only update the gradients that are relevant for pooling.
        d_pooling = backprop_pooling([d_outputs[i] for i, _ in aligned_outputs])
        for (i, _), d_pooling_i in zip(aligned_outputs, d_pooling):
            d_outputs[i] = d_pooling_i

        to_zip = (trf_datas, d_outputs, backprops_alignment)
        assert all_equal(len(x) for x in to_zip)  # type: ignore
        zipped = zip(*to_zip)
        for trf_data, d_output, get_d_src in zipped:
            if "last_hidden_state" not in trf_data.model_output:
                # This gradient belongs to a zero-length doc and must be ignored as it doesn't have a corresponding
                # output from the transformer model (due to empty documents being skipped during the span generation
                # stage in the forward pass).
                assert len(d_output) == 0
                assert get_d_src is None
                continue

            assert get_d_src is not None
            d_model_output = ModelOutput(
                last_hidden_state=model.ops.alloc(
                    trf_data.model_output.last_hidden_state.shape,  # type: ignore
                    dtype=trf_data.model_output.last_hidden_state.dtype,  # type: ignore
                )
            )
            d_src = get_d_src(d_output)
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


def concat_pooling_forward(
    pooling: Model[Ragged, Floats2d], X: List[Ragged], is_train: bool
):
    xp = pooling.ops.xp

    datas = []
    lens = []
    doc_lens = []
    for X_doc_data in X:
        datas.append(X_doc_data.dataXd)
        lens.append(X_doc_data.lengths)
        doc_lens.append(len(X_doc_data.lengths))

    X_flat = Ragged(xp.concatenate(datas, axis=0), xp.concatenate(lens, axis=0))
    Y_pooled, pooling_backprop = pooling(X_flat, is_train)
    Y = xp.split(Y_pooled, numpy.cumsum(doc_lens)[:-1])

    def backprop(dY):
        dY_pooled_flat = xp.concatenate(dY)
        dY_flat = pooling_backprop(dY_pooled_flat).dataXd

        dY = []
        for X_doc_data in X:
            doc_unpooled_len = X_doc_data.dataXd.shape[0]
            dY.append(Ragged(dY_flat[:doc_unpooled_len], X_doc_data.lengths))
            dY_flat = dY_flat[doc_unpooled_len:]

        return dY

    return Y, backprop
