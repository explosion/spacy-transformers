from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from dataclasses import dataclass
import torch
from thinc.types import Floats3d
from thinc.api import get_array_module
from thinc.api import torch2xp, xp2torch
from spacy.tokens import Span

from ._align import align_docs

BatchEncoding = Dict


@dataclass
class TransformerData:
    """Transformer tokens and outputs for one Doc object."""

    spans: List[Tuple[int, int]]
    tokens: BatchEncoding
    tensors: List[Floats3d]
    align: List[List[Tuple[int, int]]]

    @classmethod
    def empty(cls) -> "TransformerData":
        return cls(tokens={}, tensors=[], spans=[], align=[],)

    @property
    def width(self) -> int:
        for tensor in reversed(self.tensors):
            if len(tensor.shape) == 3:
                return tensor.shape[-1]
        else:
            raise ValueError("Cannot find last hidden layer")


@dataclass
class FullTransformerBatch:
    spans = List[Span]
    tokens: BatchEncoding
    tensors: List[torch.Tensor]
    _doc_data: Optional[List[TransformerData]]

    def __init__(
        self, spans: List[Span], tokens: BatchEncoding, tensors: List[torch.Tensor]
    ):
        self.spans = spans
        self.tokens = tokens
        self.tensors = tensors
        self._doc_data = None

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
        )

    def split_by_doc(self) -> List["TransformerData"]:
        """Split a TransformerData that represents a batch into a list with
        one TransformerData per Doc.
        """
        spans_by_doc = defaultdict(list)
        for span in self.spans:
            key = id(span.doc)
            spans_by_doc[key].append(span)
        outputs = []
        start = 0
        alignments = align_docs(self.spans, self.tokens["offset_mapping"])
        for doc_spans, align in zip(spans_by_doc.values(), alignments):
            end = start + len(doc_spans)
            tokens = self.slice_tokens(self.tokens, start, end)
            outputs.append(
                TransformerData(
                    spans=[(span.start, span.end) for span in doc_spans],
                    tensors=[torch2xp(t[start:end]) for t in self.tensors],
                    tokens=tokens,
                    align=[[(i-start, j) for i, j in idx] for idx in align]
                )
            )
            start += len(doc_spans)
        return outputs

    @staticmethod
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


