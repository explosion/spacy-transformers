from typing import Dict, List, Union
import torch
import copy
from transformers.file_utils import ModelOutput

from spacy.tokens import Doc
from thinc.api import Model

from ..data_classes import FullTransformerBatch
from ..layers.hf_shim import HFObjects
from ..span_getters import get_doc_spans
from ..layers.transformer_model import forward as transformer_forward


class DummyTokenizer:
    def __init__(self):
        self.str2int = {}
        self.int2str = {}
        self.start_symbol = "<s>"
        self.end_symbol = "</s>"
        self.model_max_length = 512
        self.pad_token = "[PAD]"

    @property
    def all_special_tokens(self):
        return [self.start_symbol, self.end_symbol]

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        max_length=None,
        stride: int = 0,
        truncation_strategy="longest_first",
        padding=False,
        truncation=False,
        is_pretokenized=False,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_masks=False,
        return_offsets_mapping=False,
        return_length=False,
    ):
        output: Dict = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "offset_mapping": [],
        }  # type: ignore

        for text in texts:
            words, offsets, mask, type_ids = self._tokenize(text)
            ids = self._encode_words(words)
            output["input_ids"].append(ids)
            output["attention_mask"].append(mask)
            output["token_type_ids"].append(type_ids)
            output["offset_mapping"].append(offsets)
        if padding:
            output = self._pad(output)
        if return_tensors == "pt":
            output["input_ids"] = torch.tensor(output["input_ids"])  # type: ignore
            output["attention_mask"] = torch.tensor(output["attention_mask"])  # type: ignore
            output["token_type_ids"] = torch.tensor(output["token_type_ids"])  # type: ignore
        if return_length:
            output["length"] = torch.tensor([len(x) for x in output["input_ids"]])  # type: ignore
        return output

    def convert_ids_to_tokens(self, ids: Union[List[int], torch.Tensor]) -> List[str]:
        return [self.int2str[int(id_)] for id_ in ids]  # type: ignore

    def _pad(self, batch):
        batch = copy.deepcopy(batch)
        longest = max(len(ids) for ids in batch["input_ids"])
        for i in range(len(batch["input_ids"])):
            length = len(batch["input_ids"][i])
            difference = longest - length
            batch["attention_mask"][i] = [1] * length + [0] * difference
            batch["input_ids"][i].extend([0] * difference)
            batch["token_type_ids"][i].extend([2] * difference)
            batch["offset_mapping"][i].extend([None] * difference)
        return batch

    def _tokenize(self, text):
        offsets = []
        start = 0
        for i, char in enumerate(text):
            if char == " ":
                offsets.append((start, i))
                start = i + 1
        if start < len(text):
            offsets.append((start, len(text)))
        words = [text[start:end] for start, end in offsets]
        type_ids = [0] + [1] * len(words) + [0]
        words = [self.start_symbol] + words + [self.end_symbol]
        offsets = [None] + offsets + [None]
        mask = [1] * len(words)
        return words, offsets, mask, type_ids

    def _encode_words(self, words):
        ids = []
        for word in words:
            if word not in self.str2int:
                self.int2str[len(self.str2int)] = word
                self.str2int[word] = len(self.str2int)
            ids.append(self.str2int[word])
        return ids


def DummyTransformerModel(width: int, depth: int):
    def _forward(model, tokens, is_train):
        width = model.attrs["width"]
        depth = model.attrs["depth"]
        shape = (depth, tokens.input_ids.shape[0], tokens.input_ids.shape[1], width)
        tensors = torch.zeros(*shape)
        return ModelOutput(last_hidden_state=tensors), lambda d_tensors: tokens

    return Model(
        "dummy-transformer",
        _forward,
        attrs={"width": width, "depth": depth},
    )


def DummyTransformer(
    depth: int = 2, width: int = 4, get_spans=get_doc_spans
) -> Model[List[Doc], FullTransformerBatch]:
    """Create a test model that produces a FullTransformerBatch object."""
    hf_model = HFObjects(DummyTokenizer(), None)

    return DummyModel(
        "dummy-transformer",
        transformer_forward,
        layers=[DummyTransformerModel(width=width, depth=depth)],
        attrs={
            "get_spans": get_spans,
            "hf_model": hf_model,
            "grad_factor": 1.0,
            "flush_cache_chance": 0.0,
            "transformer_config": {},
        },
        dims={"nO": width},
    )


class DummyModel(Model):
    @property
    def tokenizer(self):
        return DummyTokenizer()

    @property
    def transformer(self):
        return None

    @property
    def tokenizer_config(self):
        return {}

    @property
    def transformer_config(self):
        return {}
