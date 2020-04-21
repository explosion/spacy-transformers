from typing import Tuple, Callable, List, Optional

from spacy.util import registry
from spacy.pipeline.tok2vec import Tok2VecListener


@registry.architectures.register("spacy.Transformer.v1")
def Transformer(name: str) -> Model[TokensPlus, List[Floats2d]]:
    return PyTorchWrapper(
        AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_inputs,
        convert_outputs=convert_transformer_outputs,
    )


def convert_transformer_inputs(model, tokens: TokensPlus, is_train):
    kwargs = {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "token_type_ids": tokens.token_type_ids,
    }
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def convert_transformer_outputs(model, inputs_outputs, is_train):
    layer_inputs, torch_outputs = inputs_outputs
    torch_tokvecs: torch.Tensor = torch_outputs[0]
    # Free the memory as soon as we can
    torch_outputs = None
    lengths = list(layer_inputs.input_len)
    tokvecs: List[Floats2d] = model.ops.unpad(torch2xp(torch_tokvecs), lengths)
    # Remove the BOS and EOS markers.
    tokvecs = [arr[1:-1] for arr in tokvecs]

    def backprop(d_tokvecs: List[Floats2d]) -> ArgsKwargs:
        # Restore entries for bos and eos markers.
        row = model.ops.alloc2f(1, d_tokvecs[0].shape[1])
        d_tokvecs = [model.ops.xp.vstack((row, arr, row)) for arr in d_tokvecs]
        return ArgsKwargs(
            args=(torch_tokvecs,),
            kwargs={"grad_tensors": xp2torch(model.ops.pad(d_tokvecs))},
        )

    return tokvecs, backprop
