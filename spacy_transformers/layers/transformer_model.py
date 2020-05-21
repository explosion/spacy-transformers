from typing import List, Tuple, Callable
import torch
from transformers import AutoModel, AutoTokenizer
from spacy.tokens import Doc
from thinc.api import PyTorchWrapper, Model, CupyOps
from thinc.types import ArgsKwargs
from spacy.util import registry

from .util import huggingface_tokenize
from .util import BatchEncoding, FullTransformerBatch, TransformerData
from ._align import get_alignment


@registry.architectures.register("spacy-transformers.unloaded_transformer.v1")
def unloaded_transformer(get_spans: Callable, config) -> Model[List[Doc], TransformerData]:
    return Model(
        "transformer",
        forward,
        layers=[],
        attrs={
            "tokenizer": None,
            "get_spans": get_spans,
            "load": partial(load_transformer, get_spans=get_spans, config=config)
        },
        dims={"nO": None},
    )


@registry.architectures.register("spacy-transformers.load_transformer.v1")
def load_transformer(
    load_from: Union[str, Path], get_spans: Callable, config: dict
) -> Model[List[Doc], TransformerData]:
    transformer = AutoModel.from_pretrained(load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from, **config)
    if isinstance(model.ops, CupyOps):
        transformer.cuda()
    return TransformerModel(transformer, tokenizer, get_spans)


@registry.architectures.register("spacy.TransformerModel.v1")
def TransformerModel(
    transformer, tokenizer, get_spans: Callable
) -> Model[List[Doc], TransformerData]:
    return Model(
        "transformer",
        forward,
        layers=[
            PyTorchWrapper(
                transformer, 
                convert_inputs=_convert_transformer_inputs,
                convert_outputs=_convert_transformer_outputs,
            )
        ],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans, "load": _load},
        dims={"nO": None},
    )


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
