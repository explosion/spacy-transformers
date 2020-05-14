import numpy
from typing import List, Callable, Optional, Tuple, Dict
import torch
from dataclasses import dataclass
from spacy.tokens import Doc

from collections import defaultdict
from thinc.types import Ragged, Floats3d, FloatsXd
from thinc.api import get_array_module, registry
from thinc.api import torch2xp, xp2torch
from spacy.tokens import Span
from ._align import get_token_positions

BatchEncoding = Dict


@dataclass
class TransformerData:
    """Transformer tokens and outputs for one Doc object."""

    spans: List[Tuple[int, int]]
    tokens: BatchEncoding
    tensors: List[FloatsXd]
    align: Ragged

    @classmethod
    def empty(cls) -> "TransformerData":
        align = Ragged(numpy.zeros((0,), dtype="i"), numpy.zeros((0,), dtype="i"))
        return cls([], {}, [], align)

    @property
    def width(self) -> int:
        for tensor in reversed(self.tensors):
            if len(tensor.shape) == 3:
                return tensor.shape[-1]
        else:
            raise ValueError("Cannot find last hidden layer")


@dataclass
class FullTransformerBatch:
    spans: List[Span]
    tokens: BatchEncoding
    tensors: List[torch.Tensor]
    align: Ragged
    _doc_data: Optional[List[TransformerData]] = None

    @property
    def doc_data(self) -> List[TransformerData]:
        if self._doc_data is None:
            self._doc_data = self.split_by_doc()
        return self._doc_data

    def unsplit_by_doc(self, arrays: List[List[Floats3d]]) -> "FullTransformerBatch":
        xp = get_array_module(arrays[0][0])
        return FullTransformerBatch(
            spans=self.spans,
            tokens=self.tokens,
            tensors=[xp2torch(xp.vstack(x)) for x in transpose_list(arrays)],
            align=self.align,
        )

    def split_by_doc(self) -> List["TransformerData"]:
        """Split a TransformerData that represents a batch into a list with
        one TransformerData per Doc.
        """
        spans_by_doc = defaultdict(list)
        for span in self.spans:
            spans_by_doc[id(span.doc)].append(span)
        token_positions = get_token_positions(self.spans)
        outputs = []
        start = 0
        for doc_spans in spans_by_doc.values():
            start_i = token_positions[doc_spans[0][0]]
            end_i = token_positions[doc_spans[-1][-1]] + 1
            end = start + len(doc_spans)
            outputs.append(
                TransformerData(
                    spans=[(span.start, span.end) for span in doc_spans],
                    tokens=slice_tokens(self.tokens, start, end),
                    tensors=[torch2xp(t[start:end]) for t in self.tensors],  # type: ignore
                    align=self.align[start_i:end_i],
                )
            )
            start += len(doc_spans)
        return outputs


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerData.empty())


@registry.layers("spacy-transformers.strided_spans.v1")
def configure_strided_spans(window: int, stride: int) -> Callable:
    def get_strided_spans(docs):
        spans = []
        for doc in docs:
            start = 0
            for i in range(len(doc) // stride):
                spans.append(doc[start : start + window])
                start += stride
            if start == 0 or (start + window) < len(doc):
                spans.append(doc[start:])
        return spans

    return get_strided_spans


@registry.layers("spacy-transformers.get_sent_spans.v1")
def configure_get_sent_spans():
    def get_sent_spans(docs):
        sents = []
        for doc in docs:
            sents.extend(doc.sents)
        return sents

    return get_sent_spans


@registry.layers("spacy-transformers.get_doc_spans.v1")
def configure_get_doc_spans():
    def get_doc_spans(docs):
        return [doc[:] for doc in docs]

    return get_doc_spans


def huggingface_tokenize(tokenizer, texts) -> BatchEncoding:
    token_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=False,
        return_tensors="pt",
        return_token_type_ids=None,  # Sets to model default
        pad_to_max_length=True,
    )
    token_data["input_texts"] = [
        tokenizer.convert_ids_to_tokens(list(ids)) for ids in token_data["input_ids"]
    ]
    return token_data


def slice_hf_tokens(inputs: BatchEncoding, start: int, end: int) -> BatchEncoding:
    output = {}
    for key, value in inputs.items():
        if not hasattr(value, "__getitem__"):
            output[key] = value
        else:
            output[key] = value[start:end]
    return output


def find_last_hidden(tensors) -> int:
    for i, tensor in reversed(list(enumerate(tensors))):
        if len(tensor.shape) == 3:
            return i
    else:
        raise ValueError("No 3d tensors")


def null_annotation_setter(docs: List[Doc], trf_data: FullTransformerBatch) -> None:
    """Set no additional annotations on the Doc objects."""
    pass


def slice_tokens(inputs: BatchEncoding, start: int, end: int) -> BatchEncoding:
    output = {}
    for key, value in inputs.items():
        if not hasattr(value, "__getitem__"):
            output[key] = value
        else:
            output[key] = value[start:end]
    return output


def transpose_list(nested_list):
    output = []
    for i, entry in enumerate(nested_list):
        while len(output) < len(entry):
            output.append([None] * len(nested_list))
        for j, x in enumerate(entry):
            output[j][i] = x
    return output
