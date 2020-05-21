from typing import List, Tuple, Callable
import torch
from spacy.tokens import Doc
from thinc.api import PyTorchWrapper, Model
from thinc.types import ArgsKwargs
from transformers.tokenization_utils import BatchEncoding

from ..data_classes import FullTransformerBatch, TransformerData
from ..util import huggingface_tokenize, huggingface_from_pretrained
from ..align import get_alignment


def TransformerModel(source: str, get_spans: Callable, config: dict) -> Model[List[Doc], TransformerData]:
    return Model(
        "transformer",
        forward,
        init=init,
        layers=[],
        dims={"nO": None},
        attrs={
            "tokenizer": None,
            "get_spans": get_spans,
            "source": source,
            "config": config,
            "set_transformer": set_pytorch_transformer,
            "has_transformer": False
        }
    )


def set_pytorch_transformer(model, transformer):
    if model.attrs["has_transformer"]:
        raise ValueError("Cannot set second transformer.")
    model.layers.append(
        PyTorchWrapper(
            transformer,
            convert_inputs=_convert_transformer_inputs,
            convert_outputs=_convert_transformer_outputs,
        )
    )
    model.attrs["has_transformer"] = True


def init(model, X=None, Y=None):
    if model["has_transformer"]:
        return
    source = model.attrs["source"]
    config = model.attrs["config"]
    tokenizer, transformer = huggingface_from_pretrained(source, config)
    model.attrs["tokenizer"] = tokenizer
    model.attrs["set_transformer"](model, transformer)
    for layer in model.layers:
        layer.initialize()


def forward(
    model: Model, docs: List[Doc], is_train: bool
) -> Tuple[FullTransformerBatch, Callable]:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    spans = get_spans(docs)
    span_docs = {id(span.doc) for span in spans}
    for doc in docs:
        if id(doc) not in span_docs:
            raise ValueError(doc.text)
    token_data = huggingface_tokenize(tokenizer, [span.text for span in spans])
    tensors, bp_tensors = transformer(token_data, is_train)
    output = FullTransformerBatch(
        spans=spans,
        tokens=token_data,
        tensors=tensors,
        align=get_alignment(spans, token_data["input_texts"]),
    )

    def backprop_transformer(d_output: FullTransformerBatch) -> List[Doc]:
        _ = bp_tensors(d_output.tensors)
        return docs

    return output, backprop_transformer


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
