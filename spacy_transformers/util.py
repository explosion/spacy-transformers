import numpy
import catalogue
from typing import List, Callable, cast
from typing import Optional, Tuple, List, Dict
import torch
from dataclasses import dataclass
from spacy.tokens import Doc

from collections import defaultdict
from thinc.types import Ragged, Floats2d, Floats3d, FloatsXd
from thinc.api import get_array_module
from thinc.api import torch2xp, xp2torch
from thinc.types import Decorator
from spacy.tokens import Span
from ._align import BatchAlignment


# TODO: How should we register this?
spanners = catalogue.create("spacy-transformers", "spanners")


BatchEncoding = Dict


@dataclass
class TransformerData:
    """Transformer tokens and outputs for one Doc object."""

    spans: List[Tuple[int, int]]
    tokens: BatchEncoding
    tensors: List[FloatsXd]
    trf2tok: Ragged
    tok2trf: Ragged

    @classmethod
    def empty(cls) -> "TransformerData":
        a2b = numpy.zeros((0,), dtype="i")
        b2a = numpy.zeros((0,), dtype="i")
        return cls([], {}, [], a2b, b2a)

    @property
    def width(self) -> int:
        for tensor in reversed(self.tensors):
            if len(tensor.shape) == 3:
                return tensor.shape[-1]
        else:
            raise ValueError("Cannot find last hidden layer")

    def align_to_tokens(self, ops, wp: Floats2d) -> Tuple[Ragged, Callable]:
        aligned = Ragged(wp[self.trf2tok.data], self.trf2tok.lengths)

        def backprop_tok_alignment(d_aligned: Ragged) -> Floats3d:
            d_wp = ops.alloc2f(len(d_aligned), d_aligned.data.shape[1])
            ops.scatter_add(d_wp, self.trf2tok.data, d_aligned.data)
            return d_wp

        return aligned, backprop_tok_alignment

    def align_to_transformer(self, ops, tok: Floats2d) -> Tuple[Ragged, Callable]:
        aligned = Ragged(tok[self.tok2trf.data], self.tok2trf.lengths)

        def backprop_wp_alignment(d_aligned: Ragged) -> Floats2d:
            d_tok = ops.alloc2f(len(d_aligned), d_aligned.data.shape[1])
            ops.scatter_add(d_tok, self.tok2trf.data, d_aligned.data)
            return d_tok

        return aligned, backprop_wp_alignment


@dataclass
class FullTransformerBatch:
    spans: List[Span]
    tokens: BatchEncoding
    tensors: List[torch.Tensor]
    align: BatchAlignment
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
            align=self.align
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
        for doc_spans in spans_by_doc.values():
            end = start + len(doc_spans)
            trf2tok, tok2trf = self.align.slice(start, end)

            torch_slices = [t[start:end] for t in self.tensors]
            outputs.append(
                TransformerData(
                    spans=[(span.start, span.end) for span in doc_spans],
                    tokens=slice_tokens(self.tokens, start, end),
                    tensors=[torch2xp(t) for t in torch_slices], # type: ignore
                    trf2tok=trf2tok,
                    tok2trf=tok2trf
                )
            )
            start += len(doc_spans)
        return outputs


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerData.empty())



@spanners("spacy-transformers.strided_spans.v1")
def configure_strided_spans(window: int, stride: int) -> Callable:
    def get_strided_spans(docs):
        spans = []
        for doc in docs:
            start = 0
            for i in range(len(doc) // stride):
                spans.append(doc[start : start + window])
                start += stride
            if start < len(doc):
                spans.append(doc[start : ])
        return spans
    return get_strided_spans


@spanners("spacy-transformers.get_sent_spans.v1")
def configure_get_sent_spans():
    def get_sent_spans(docs):
        sents = []
        for doc in docs:
            sents.extend(doc.sents)
        return sents
    return get_sent_spans


@spanners("spacy-transformers.get_doc_spans.v1")
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
    )
    token_data["input_texts"] = [tokenizer.convert_ids_to_tokens(list(ids)) for ids in token_data["input_ids"]]
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
