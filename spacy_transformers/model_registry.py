from typing import Tuple, Callable, List, Optional

from spacy.util import registry
from thinc.api import PyTorchWrapper, Model, torch2xp, xp2torch
from .types import TransformerOutput, TokensPlus


@thinc.registry.layers("spacy.Transformer.v1")
def Transformer(
    transformer,
    tokenizer,
) -> Model[List[Span], TransformerOutput]:
    wrapper = PyTorchWrapper(
        transformer, # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_inputs,
    )
    return Model(
        "transformer",
        transformer_forward,
        layers=[wrapper],
        attrs={"tokenizer": tokenizer}
    )


def convert_transformer_inputs(model, tokens: TokensPlus, is_train):
    kwargs = {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "token_type_ids": tokens.token_type_ids,
    }
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def transformer_forward(model: Model, spans: List[Span], is_train: bool) -> TransformerOutput:
    tokenizer = model.attrs["tokenizer"]
    transformer = model.layers[0]
    
    token_data = tokenizer.batch_encode_plus(
        [span.text for span in spans],
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_masks=True,
        return_input_lengths=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    tokens = TokensPlus(**token_data)

    tensors, bp_tensors = transformer(tokens, is_train)
    output = TransformerOutput(tokens=tokens, tensors=tensors, spans=spans)

    def backprop_sentence_transformer(d_output):
        _ = bp_tensors(d_output.tensors)
        return spans

    return output
