from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from spacy.tokens import Doc
from thinc.api import PyTorchWrapper, Model
from thinc.types import ArgsKwargs
from spacy.util import registry

from .types import BatchEncoding, FullTransformerBatch, TransformerData
from .util import get_doc_spans, huggingface_tokenize


@registry.architectures.register("spacy.TransformerByName.v1")
def TransformerModelByName(
    name: str, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerData]:
    transformer = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return TransformerModel(transformer, tokenizer, get_spans=get_spans)


@registry.architectures.register("spacy.TransformerModel.v1")
def TransformerModel(
    transformer, tokenizer, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerData]:
    wrapper = PyTorchTransformer(transformer)
    width = transformer.config.dim
    return Model(
        "transformer",
        forward,
        layers=[wrapper],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans},
        dims={"nO": width}
    )


def forward(model: Model, docs: List[Doc], is_train: bool) -> FullTransformerBatch:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    spans = get_spans(docs)
    token_data = huggingface_tokenize(tokenizer, [span.text for span in spans])
    tensors, bp_tensors = transformer(token_data, is_train)
    output = FullTransformerBatch(spans=spans, tokens=token_data, tensors=tensors)

    def backprop_transformer(d_output: FullTransformerBatch):
        _ = bp_tensors(d_output.tensors)
        return docs

    return output, backprop_transformer


def PyTorchTransformer(transformer):
    return PyTorchWrapper(
        transformer,  # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=_convert_transformer_inputs,
        convert_outputs=_convert_transformer_outputs,
    )


def _convert_transformer_inputs(model, tokens: BatchEncoding, is_train):
    # Adapter for the PyTorchWrapper. See https://thinc.ai/docs/usage-frameworks
    kwargs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }
    if "token_type_ids" in tokens:
        kwargs["token_type_ids"] = tokens["token_type_ids"]
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def _convert_transformer_outputs(model, inputs_outputs, is_train):
    _, tensors = inputs_outputs

    def backprop(d_tensors: List[torch.Tensor]) -> ArgsKwargs:
        return ArgsKwargs(args=(tensors,), kwargs={"grad_tensors": d_tensors})

    return tensors, backprop
