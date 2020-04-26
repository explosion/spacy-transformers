from typing import Optional, Dict, List
import torch

from spacy.tokens import Doc
from thinc.api import Model

from ..types import TransformerOutput
from ..model_wrapper import get_doc_spans


class DummyTokenizer:
    def __init__(self):
        self.str2int = {}
        self.int2str = {}
        self.start_symbol = "<s>"
        self.end_symbol = "</s>"

    def batch_encode_plus(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_masks: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_masks: bool = False,
        return_offsets_mapping: bool = False,
        return_lengths: bool = False,
    ) -> Dict:
        output = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "offset_mapping": [],
        }

        for text in texts:
            words, offsets, mask, type_ids = self._tokenize(text)
            ids = self._encode_words(words)
            output["input_ids"].append(ids)
            output["attention_mask"].append(mask)
            output["token_type_ids"].append(type_ids)
            output["offset_mapping"].append(offsets)
        return output

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
        mask = list(range(len(words)))
        return words, offsets, mask, type_ids

    def _encode_words(self, words):
        ids = []
        for word in words:
            if word not in self.str2int:
                self.str2int[word] = len(self.str2int)
            ids.append(self.str2int[word])
        return ids


def DummyTransformer(
    depth: int = 2, width: int = 4, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    """Create a test model that produces a TransformerOutput object."""
    tokenizer = DummyTokenizer()
    return Model(
        "test-transformer",
        forward_dummy_transformer,
        attrs={
            "width": width,
            "depth": depth,
            "get_spans": get_spans,
            "tokenizer": tokenizer,
        },
    )


def forward_dummy_transformer(model, docs, is_train):
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    width = model.attrs["width"]
    depth = model.attrs["depth"]

    spans = get_spans(docs)
    tokens = tokenizer(spans)

    tensors = []
    shape = (tokens.input_ids.shape[0], tokens.input_ids.shape[1], width)
    for i in range(depth):
        tensors.append(torch.tensor(shape))

    output = TransformerOutput(tokens=tokens, tensors=tensors, spans=spans)

    def backprop(d_output: TransformerOutput):
        return docs

    return output, backprop
