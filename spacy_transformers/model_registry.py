from typing import Tuple, Callable, List, Optional

from spacy.util import registry
from thinc.api import PyTorchWrapper, Model, torch2xp, xp2torch
from .types import TransformerOutput, TokensPlus


@thinc.registry.layers("transformers_tokenizer.v1")
def TransformersTokenizer(name: str) -> Model[List[List[str]], TokensPlus]:
    def forward(
        model, texts: List[List[str]], is_train: bool
    ) -> Tuple[TokensPlus, Callable]:
        tokenizer = model.attrs["tokenizer"]
        token_data = tokenizer.batch_encode_plus(
            [(text, None) for text in texts],
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_masks=True,
            return_input_lengths=True,
            return_tensors="pt",
        )
        return TokensPlus(**token_data), lambda d_tokens: []

    return Model(
        "tokenizer", forward, attrs={"tokenizer": AutoTokenizer.from_pretrained(name)},
    )


@registry.architectures.register("spacy.Transformer.v1")
def Transformer(name: str) -> Model[TokensPlus, TransformerOutput]:
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


def convert_transformer_outputs(model, tokens_tensors, is_train):
    tokens, tensors = tokens_tensors
    output = TransformerOutput(tokens, [torch2xp(tensor) for tensor in tensors])

    def backprop(d_output: TransformerOutput) -> ArgsKwargs:
        return ArgsKwargs(
            args=(tensors,),
            kwargs={"grad_tensors": [xp2torch(x) for x in TransformerOutput.tensors]}
        )

    return output, backprop
