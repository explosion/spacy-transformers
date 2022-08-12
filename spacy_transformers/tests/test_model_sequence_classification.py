from typing import Callable
from functools import partial
import copy

import torch
from transformers import AutoModelForSequenceClassification
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import spacy
from thinc.api import Model

from spacy_transformers.data_classes import HFObjects, WordpieceBatch
from spacy_transformers.layers.hf_wrapper import HFWrapper
from spacy_transformers.layers.transformer_model import _convert_transformer_inputs
from spacy_transformers.layers.transformer_model import _convert_transformer_outputs
from spacy_transformers.layers.transformer_model import forward
from spacy_transformers.layers.transformer_model import huggingface_from_pretrained
from spacy_transformers.layers.transformer_model import huggingface_tokenize
from spacy_transformers.layers.transformer_model import set_pytorch_transformer
from spacy_transformers.span_getters import get_strided_spans


def test_model_for_sequence_classification():
    # adapted from https://github.com/KennethEnevoldsen/spacy-wrap/
    class ClassificationTransformerModel(Model):
        def __init__(
            self,
            name: str,
            get_spans: Callable,
            tokenizer_config: dict = {},
            transformer_config: dict = {},
            mixed_precision: bool = False,
            grad_scaler_config: dict = {},
        ):
            hf_model = HFObjects(None, None, None, tokenizer_config, transformer_config)
            wrapper = HFWrapper(
                hf_model,
                convert_inputs=_convert_transformer_inputs,
                convert_outputs=_convert_transformer_outputs,
                mixed_precision=mixed_precision,
                grad_scaler_config=grad_scaler_config,
                model_cls=AutoModelForSequenceClassification,
            )
            super().__init__(
                "clf_transformer",
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
            copied = ClassificationTransformerModel(self.name, self.attrs["get_spans"])
            params = {}
            for name in self.param_names:
                params[name] = self.get_param(name) if self.has_param(name) else None
            copied.params = copy.deepcopy(params)
            copied.dims = copy.deepcopy(self._dims)
            copied.layers[0] = copy.deepcopy(self.layers[0])
            for name in self.grad_names:
                copied.set_grad(name, self.get_grad(name).copy())
            return copied

    def init(model: ClassificationTransformerModel, X=None, Y=None):
        if model.attrs["has_transformer"]:
            return
        name = model.attrs["name"]
        tok_cfg = model._init_tokenizer_config
        trf_cfg = model._init_transformer_config
        hf_model = huggingface_from_pretrained(
            name, tok_cfg, trf_cfg, model_cls=AutoModelForSequenceClassification
        )
        model.attrs["set_transformer"](model, hf_model)
        tokenizer = model.tokenizer
        texts = ["hello world", "foo bar"]
        token_data = huggingface_tokenize(tokenizer, texts)
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
        model.layers[0].initialize(X=wordpieces)

    model = ClassificationTransformerModel(
        "sgugger/tiny-distilbert-classification",
        get_spans=partial(get_strided_spans, window=128, stride=96),
    )
    model.initialize()

    assert isinstance(model.transformer, DistilBertForSequenceClassification)
    nlp = spacy.blank("en")
    doc = nlp.make_doc("some text")
    assert isinstance(model.predict([doc]).model_output, SequenceClassifierOutput)

    b = model.to_bytes()
    model_re = ClassificationTransformerModel(
        "sgugger/tiny-distilbert-classification",
        get_spans=partial(get_strided_spans, window=128, stride=96),
    ).from_bytes(b)
    assert isinstance(model_re.transformer, DistilBertForSequenceClassification)
    assert isinstance(model_re.predict([doc]).model_output, SequenceClassifierOutput)
    assert torch.equal(
        model.predict([doc]).model_output.logits,
        model_re.predict([doc]).model_output.logits,
    )
    # Note that model.to_bytes() != model_re.to_bytes(), but this is also not
    # true for the default models.
