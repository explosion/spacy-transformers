import copy
from typing import List, Tuple, Callable
from transformers.file_utils import ModelOutput

from spacy.tokens import Doc
from thinc.api import Model, xp2torch
from thinc.types import ArgsKwargs

import logging

from ..data_classes import FullTransformerBatch, WordpieceBatch
from ..util import huggingface_tokenize, huggingface_from_pretrained
from ..util import maybe_flush_pytorch_cache
from ..util import log_gpu_memory, log_batch_size
from ..layers._util import replace_listener, replace_listener_cfg
from ..truncate import truncate_oversize_splits
from ..align import get_alignment
from .hf_wrapper import HFWrapper
from .hf_shim import HFObjects


class TransformerModel(Model):
    def __init__(
        self,
        name: str,
        get_spans: Callable,
        tokenizer_config: dict = {},
        transformer_config: dict = {},
        mixed_precision: bool = False,
        grad_scaler_config: dict = {},
    ):
        """
        get_spans (Callable[[List[Doc]], List[Span]]):
            A function to extract spans from the batch of Doc objects.
            This is used to manage long documents, by cutting them into smaller
            sequences before running the transformer. The spans are allowed to
            overlap, and you can also omit sections of the Doc if they are not
            relevant.
        tokenizer_config (dict): Settings to pass to the transformers tokenizer.
        transformer_config (dict): Settings to pass to the transformers forward pass.
        """
        hf_model = HFObjects(None, None, tokenizer_config, transformer_config)
        wrapper = HFWrapper(
            hf_model,
            convert_inputs=_convert_transformer_inputs,
            convert_outputs=_convert_transformer_outputs,
            mixed_precision=mixed_precision,
            grad_scaler_config=grad_scaler_config,
        )
        super().__init__(
            "transformer",
            forward,
            init=init,
            layers=[wrapper],
            dims={"nO": None},
            attrs={
                "get_spans": get_spans,
                "name": name,
                "set_transformer": set_pytorch_transformer,
                "has_transformer": False,
                "flush_cache_chance": 0.0,
                "replace_listener": replace_listener,
                "replace_listener_cfg": replace_listener_cfg,
            },
        )

    @property
    def tokenizer(self):
        return self.layers[0].shims[0]._hfmodel.tokenizer

    @property
    def transformer(self):
        return self.layers[0].shims[0]._hfmodel.transformer

    @property
    def _init_tokenizer_config(self):
        return self.layers[0].shims[0]._hfmodel._init_tokenizer_config

    @property
    def _init_transformer_config(self):
        return self.layers[0].shims[0]._hfmodel._init_transformer_config

    def copy(self):
        """
        Create a copy of the model, its attributes, and its parameters. Any child
        layers will also be deep-copied. The copy will receive a distinct `model.id`
        value.
        """
        copied = TransformerModel(self.name, self.attrs["get_spans"])
        params = {}
        for name in self.param_names:
            params[name] = self.get_param(name) if self.has_param(name) else None
        copied.params = copy.deepcopy(params)
        copied.dims = copy.deepcopy(self._dims)
        copied.layers[0] = copy.deepcopy(self.layers[0])
        for name in self.grad_names:
            copied.set_grad(name, self.get_grad(name).copy())
        return copied


def set_logger(model, out_file):
    """Add a logger that will log memory usage to the given file.

    Used to debug OOM errors.
    """
    logging.basicConfig(
        level="INFO", format="%(asctime)s:%(levelname)s: %(message)s", stream=out_file
    )
    model.attrs["logger"] = logging.getLogger(__name__)


def set_pytorch_transformer(model, transformer, tokenizer):
    if model.attrs["has_transformer"]:
        raise ValueError("Cannot set second transformer.")
    model.layers[0].shims[0]._model = transformer
    model.layers[0].shims[0]._hfmodel.transformer = transformer
    model.layers[0].shims[0]._hfmodel.tokenizer = tokenizer
    model.attrs["has_transformer"] = True
    model.set_dim("nO", transformer.config.hidden_size)


def init(model: Model, X=None, Y=None):
    if model.attrs["has_transformer"]:
        return
    name = model.attrs["name"]
    tok_cfg = model._init_tokenizer_config
    trf_cfg = model._init_transformer_config
    tokenizer, transformer = huggingface_from_pretrained(name, tok_cfg, trf_cfg)
    model.attrs["set_transformer"](model, transformer, tokenizer)
    tokenizer = model.tokenizer
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
        token_data = huggingface_tokenize(tokenizer, [span.text for span in flat_spans])
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
        align = get_alignment(
            flat_spans, wordpieces.strings, tokenizer.all_special_tokens
        )
        wordpieces, align = truncate_oversize_splits(
            wordpieces, align, tokenizer.model_max_length
        )
    else:
        texts = ["hello world", "foo bar"]
        token_data = huggingface_tokenize(tokenizer, texts)
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
    model.layers[0].initialize(X=wordpieces)
    model_output = model.layers[0].predict(wordpieces)
    model.set_dim("nO", model_output.last_hidden_state.shape[-1])


def forward(
    model: TransformerModel, docs: List[Doc], is_train: bool
) -> Tuple[FullTransformerBatch, Callable]:
    tokenizer = model.tokenizer
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
    align = get_alignment(flat_spans, wordpieces.strings, tokenizer.all_special_tokens)
    wordpieces, align = truncate_oversize_splits(
        wordpieces, align, tokenizer.model_max_length
    )
    model_output, bp_tensors = transformer(wordpieces, is_train)
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "after forward")
    output = FullTransformerBatch(
        spans=nested_spans,
        wordpieces=wordpieces,
        model_output=model_output,
        align=align,
    )
    if "logger" in model.attrs:
        log_gpu_memory(model.attrs["logger"], "return from forward")

    def backprop_transformer(d_output: FullTransformerBatch) -> List[Doc]:
        if "logger" in model.attrs:
            log_gpu_memory(model.attrs["logger"], "Begin backprop")
        _ = bp_tensors(d_output.model_output)
        if "logger" in model.attrs:
            log_gpu_memory(model.attrs["logger"], "After backprop")
        return docs

    return output, backprop_transformer


def _convert_transformer_inputs(model, wps: WordpieceBatch, is_train):
    # Adapter for the HFWrapper. See https://thinc.ai/docs/usage-frameworks

    hf_device = model.shims[0]._hfmodel.transformer.device
    kwargs = {
        # Note: remove conversion to long when PyTorch >= 1.8.0.
        "input_ids": xp2torch(wps.input_ids).long().to(device=hf_device),
        "attention_mask": xp2torch(wps.attention_mask).to(device=hf_device),
    }
    if wps.token_type_ids is not None:
        kwargs["token_type_ids"] = (
            xp2torch(wps.token_type_ids).long().to(device=hf_device)
        )
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def _convert_transformer_outputs(model, inputs_outputs, is_train):
    _, model_output = inputs_outputs

    def backprop(d_model_output: ModelOutput) -> ArgsKwargs:
        return ArgsKwargs(
            args=(model_output.last_hidden_state,),
            kwargs={"grad_tensors": d_model_output.values()},
        )

    return model_output, backprop
