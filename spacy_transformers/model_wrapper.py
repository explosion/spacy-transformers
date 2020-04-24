from typing import Tuple, Callable, List, Optional

import torch
from spacy.tokens import Span, Doc
from spacy.util import registry
import thinc
from thinc.api import PyTorchWrapper, Model, torch2xp, xp2torch, ArgsKwargs
from thinc.types import FloatsXd
from transformers import AutoModel, AutoTokenizer

from .types import TransformerOutput, TokensPlus


def get_doc_spans(docs: List[Doc]) -> List[Span]:
    return [doc[:] for doc in docs]


def get_sents(docs: List[Doc]) -> List[Span]:
    spans = []
    for doc in docs:
        spans.extend(doc.sents)
    return spans


@thinc.registry.layers("spacy.TransformerByName.v1")
def TransformerByName(
    name: str, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    transformer = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    wrapper = PyTorchWrapper(
        transformer,  # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=_convert_transformer_inputs,
    )
    return Model(
        "transformer",
        transformer_forward,
        layers=[wrapper],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans},
    )


@thinc.registry.layers("spacy.Transformer.v1")
def Transformer(
    transformer, tokenizer, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    wrapper = PyTorchWrapper(
        transformer,  # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=_convert_transformer_inputs,
    )
    return Model(
        "transformer",
        transformer_forward,
        layers=[wrapper],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans},
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


def transformer_forward(
    model: Model, docs: List[Doc], is_train: bool
) -> TransformerOutput:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    spans = get_spans(docs)

    token_data = tokenizer.batch_encode_plus(
        [span.text for span in spans],
        add_special_tokens=True,
        return_attention_mask=True,
        return_lengths=True,
        return_offsets_mapping=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors=None,  # Work around bug :(
        return_token_type_ids=None,  # Sets to model default
    )
    # Work around https://github.com/huggingface/transformers/issues/3224
    token_data["input_ids"] = torch.tensor(token_data["input_ids"])
    token_data["attention_mask"] = torch.tensor(token_data["attention_mask"])
    if "token_type_ids" in token_data:
        token_data["token_type_ids"] = torch.tensor(token_data["token_type_ids"])
    tokens = TokensPlus(**token_data)

    tensors, bp_tensors = transformer(tokens, is_train)
    output = TransformerOutput(tokens=tokens, tensors=tensors, spans=spans)

    def backprop_transformer(d_output: TransformerOutput):
        _ = bp_tensors(d_output.tensors)
        return docs

    return output, backprop_transformer
