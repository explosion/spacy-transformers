import torch
from thinc.api import PyTorchWrapper
from thinc.types import ArgsKwargs
from ..types import TokensPlus


def PyTorchTransformer(transformer):
    return PyTorchWrapper(
        transformer,  # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=_convert_transformer_inputs,
        convert_outputs=_convert_transformer_outputs,
        dims={"nO": transformer.config.dim}
    )


def _convert_transformer_inputs(model, tokens: TokensPlus, is_train):
    # Adapter for the PyTorchWrapper. See https://thinc.ai/docs/usage-frameworks
    kwargs = {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
    }
    if tokens.token_type_ids is not None:
        kwargs["token_type_ids"] = tokens.token_type_ids
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def _convert_transformer_outputs(model, inputs_outputs, is_train):
    _, tensors = inputs_outputs

    def backprop(d_tensors: List[torch.Tensor]) -> ArgsKwargs:
        return ArgsKwargs(args=(tensors,), kwargs={"grad_tensors": d_tensors})

    return tensors, backprop
