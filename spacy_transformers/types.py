from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch
from thinc.types import Floats3d
from thinc.api import Ops, torch2xp, get_current_ops
from spacy.tokens import Span

from ._align import align_docs
from ._batch_encoding import BatchEncoding


@dataclass
class TransformerData:
    """Transformer tokens and outputs for one Doc object."""
    spans: List[Tuple[int, int]]
    tokens: BatchEncoding
    tensors: List[Floats3d]
    align: List[List[Tuple[int, int]]]

    @classmethod
    def empty(cls) -> "TransformerData":
        return cls(
            tokens=BatchEncoding(),
            tensors=[],
            spans=[],
            align=[],
        )

    @property
    def width(self) -> int:
        return self.tensors[-1].shape[-1]


@dataclass
class FullTransformerBatch:
    spans = List[Span]
    tokens: BatchEncoding
    tensors: List[torch.Tensor]
    doc_data: List[TransformerData]

    def __init__(
        self,
        spans: List[Span],
        tokens: BatchEncoding,
        tensors: List[torch.Tensor]
    ):
        self.spans = spans
        self.tokens = tokens
        self.tensors = tensors
        self.doc_data = self.split_by_doc(spans, tokens, tensors)

    @staticmethod
    def split_by_doc(spans, tokens, tensors) -> List["TransformerOutput"]:
        """Split a TransformerOutput that represents a batch into a list with
        one TransformerOutput per Doc.
        """
        spans_by_doc = defaultdict(list)
        for span in spans:
            key = id(span.doc)
            spans_by_doc[key].append(span)
        outputs = []
        start = 0
        alignments = align_docs(spans, tokens["offset_mapping"])
        for doc_spans, align in zip(spans_by_doc.values(), alignments):
            end = start + len(doc_spans)
            tokens = self.slice_tokens(tokens, start, end)
            outputs.append(
                TransformerOutput(
                    spans=[(span.start, span.end) for span in doc_spans],
                    tensors=[torch2xp(t[start : end]) for t in tensors],
                    tokens=tokens,
                    align=align
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
                output[key] = value[start : end]
        return output
