from typing import List, Tuple, Callable
import torch
from spacy.tokens import Doc
from thinc.api import PyTorchWrapper, Model, xp2torch
from thinc.types import ArgsKwargs
from transformers.tokenization_utils import BatchEncoding
import logging

from ..data_classes import FullTransformerBatch, WordpieceBatch
from ..util import huggingface_tokenize, huggingface_from_pretrained
from ..util import find_last_hidden, maybe_flush_pytorch_cache
from ..truncate import truncate_oversize_splits
from ..util import log_gpu_memory, log_batch_size
from ..align import get_alignment


def TransformerModel(
    name: str, get_spans: Callable, tokenizer_config: dict
) -> Model[List[Doc], FullTransformerBatch]:
    """
    get_spans (Callable[[List[Doc]], List[Span]]):
        A function to extract spans from the batch of Doc objects.
        This is used to manage long documents, by cutting them into smaller
        sequences before running the transformer. The spans are allowed to
        overlap, and you can also omit sections of the Doc if they are not
        relevant.
    tokenizer_config (dict): Settings to pass to the transformers tokenizer.
    """

    return Model(
        "transformer",
        forward,
        init=init,
        layers=[],
        dims={"nO": None},
        attrs={
            "tokenizer": None,
            "get_spans": get_spans,
            "name": name,
            "tokenizer_config": tokenizer_config,
            "set_transformer": set_pytorch_transformer,
            "has_transformer": False,
            "flush_cache_chance": 0.0,
        },
    )


def set_logger(model, out_file):
    """Add a logger that will log memory usage to the given file.

    Used to debug OOM errors.
    """
    logging.basicConfig(
        level="INFO", format="%(asctime)s:%(levelname)s: %(message)s", stream=out_file
    )
    model.attrs["logger"] = logging.getLogger(__name__)


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
    model.set_dim("nO", transformer.config.hidden_size)


def init(model: Model, X=None, Y=None):
    if model.attrs["has_transformer"]:
        return
    name = model.attrs["name"]
    tok_cfg = model.attrs["tokenizer_config"]
    tokenizer, transformer = huggingface_from_pretrained(name, tok_cfg)
    model.attrs["tokenizer"] = tokenizer
    model.attrs["set_transformer"](model, transformer)
    # Call the model with a batch of inputs to infer the width
    if X:
        # If we're dealing with actual texts, do the work to setup the wordpieces
        # batch properly
        docs = X
        get_spans = model.attrs["get_spans"]
        nested_spans = get_spans(docs)
        flat_spans = []
        for doc_spans in nested_spans:
            flat_spans.extend(doc_spans)
        token_data = huggingface_tokenize(
            model.attrs["tokenizer"],
            [span.text for span in flat_spans]
        )
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
        align = get_alignment(
            flat_spans,
            wordpieces.strings, model.attrs["tokenizer"].all_special_tokens
        )
        wordpieces, align = truncate_oversize_splits(
            wordpieces, align, tokenizer.model_max_length
        )
    else:
        texts = ["hello world", "foo bar"]
        token_data = huggingface_tokenize(
            model.attrs["tokenizer"],
            texts
        )
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
    model.layers[0].initialize(X=wordpieces)
    tensors = model.layers[0].predict(wordpieces)
    t_i = find_last_hidden(tensors)
    model.set_dim("nO", tensors[t_i].shape[-1])


def forward(
    model: Model, docs: List[Doc], is_train: bool
) -> Tuple[FullTransformerBatch, Callable]:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    nested_spans = get_spans(docs)
    flat_spans = []
    for doc_spans in nested_spans:
        flat_spans.extend(doc_spans)
    # Flush the PyTorch cache every so often. It seems to help with memory :(
    # This shouldn't be necessary, I'm not sure what I'm doing wrong?
    maybe_flush_pytorch_cache(chance=model.attrs.get("flush_cache_chance", 0))
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "begin forward")
    batch_encoding = huggingface_tokenize(tokenizer, [span.text for span in flat_spans])
    wordpieces = WordpieceBatch.from_batch_encoding(batch_encoding)
    if "logger" in model.attrs:
        log_batch_size(model.attrs["logger"], wordpieces, is_train)
    align = get_alignment(
        flat_spans, wordpieces.strings, model.attrs["tokenizer"].all_special_tokens
    )
    wordpieces, align = truncate_oversize_splits(
        wordpieces, align, tokenizer.model_max_length
    )
    tensors, bp_tensors = transformer(wordpieces, is_train)
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "after forward")
    output = FullTransformerBatch(
        spans=nested_spans, wordpieces=wordpieces, tensors=tensors, align=align
    )
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "return from forward")

    def backprop_transformer(d_output: FullTransformerBatch) -> List[Doc]:
        if "logger" in model.attrs:
            log_gpu_memory(model.attrs["logger"], "Begin backprop")
        _ = bp_tensors(d_output.tensors)
        if "logger" in model.attrs:
            log_gpu_memory(model.attrs["logger"], "After backprop")
        return docs

    return output, backprop_transformer


def _convert_transformer_inputs(model, wps: WordpieceBatch, is_train):
    # Adapter for the PyTorchWrapper. See https://thinc.ai/docs/usage-frameworks
    kwargs = {
        "input_ids": xp2torch(wps.input_ids),
        "attention_mask": xp2torch(wps.attention_mask),
    }
    if wps.token_type_ids is not None:
        kwargs["token_type_ids"] = xp2torch(wps.token_type_ids)
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def _convert_transformer_outputs(model, inputs_outputs, is_train):
    _, tensors = inputs_outputs
    if hasattr(tensors, "to_tuple"):
        tensors = tensors.to_tuple()

    def backprop(d_tensors: List[torch.Tensor]) -> ArgsKwargs:
        return ArgsKwargs(args=(tensors,), kwargs={"grad_tensors": d_tensors})

    return tensors, backprop
