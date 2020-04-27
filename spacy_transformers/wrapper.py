from typing import List, Optional, Dict
import torch
from transformers import AutoModel, AutoTokenizer
from spacy.tokens import Token, Span, Doc
import thinc
from thinc.api import PyTorchWrapper, Model
from thinc.types import ArgsKwargs
from spacy.util import registry

from .types import TokensPlus, TransformerOutput
from .util import get_doc_spans, get_sent_spans


@registry.architectures.register("spacy.TransformerByName.v1")
def TransformerModelByName(
    name: str, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    transformer = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return TransformerModel(transformer, tokenizer, get_spans=get_spans)


@registry.architectures.register("spacy.TransformerModel.v1")
def TransformerModel(
    transformer, tokenizer, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    wrapper = PyTorchTransformer(transformer)
    return Model(
        "transformer",
        forward,
        layers=[wrapper],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans},
    )


def forward(
    model: Model, docs: List[Doc], is_train: bool
) -> TransformerOutput:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    spans = get_spans(docs)
    token_data = tokenizer.batch_encode_plus(
        [span.text for span in spans],
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=False,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",  
        return_token_type_ids=None,  # Sets to model default
    )
    # Work around https://github.com/huggingface/transformers/issues/3224
    extra_token_data = tokenizer.batch_encode_plus(
        [span.text for span in spans],
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors=None,  
        return_token_type_ids=None,  # Sets to model default
    )
    # There seems to be some bug where it's flattening single-entry batches?
    if len(spans) == 1:
        token_data["offset_mapping"] = [extra_token_data["offset_mapping"]]
    else:
        token_data["offset_mapping"] = extra_token_data["offset_mapping"]
    tokens = TokensPlus(**token_data)

    tensors, bp_tensors = transformer(tokens, is_train)
    output = TransformerOutput(
        tokens=tokens, tensors=tensors, spans=spans, ops=transformer.ops
    )

    def backprop_transformer(d_output: TransformerOutput):
        _ = bp_tensors(d_output.tensors)
        return docs

    return output, backprop_transformer


def PyTorchTransformer(transformer):
    return PyTorchWrapper(
        transformer,  # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=_convert_transformer_inputs,
        convert_outputs=_convert_transformer_outputs,
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
