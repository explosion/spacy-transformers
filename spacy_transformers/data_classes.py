from dataclasses import dataclass
from typing import Optional, List
import torch
import numpy
from transformers.tokenization_utils import BatchEncoding
from thinc.types import Ragged, Floats3d, FloatsXd, Ints2d
from thinc.api import get_array_module, xp2torch, torch2xp
from spacy.tokens import Span

from .util import transpose_list
from .align import get_token_positions


@dataclass
class WordpieceBatch:
    """Holds data from the transformers BatchEncoding class.

    We would have preferred to use the BatchEncoding class directly, but
    there's a few problems with that.
    
    1. Some BatchEncoding functionality requires the tokenizers.Encoding object,
        and it's impossible for us to create or manipulate that object. This means
        we can't really create BatchEncoding objects, which limits what we can do.
    2. We want some semantic differences, for instance the "lengths" data in the
        BatchEncoding is useless when the inputs are padded. We want it to tell
        us the *unpadded* lengths.
    3. We want typed attributes, so that we can type-check properly.
    4. We prefer to have numpy/cupy arrays rather than torch arrays.
    5. The API around the BatchEncoding object has been changing a lot, so we
        want to minimize the places where we touch it.
    """

    strings: List[List[str]]
    input_ids: Ints2d
    attention_mask: Floats3d
    lengths: List[int]
    token_type_ids: Optional[Ints2d]

    def __len__(self) -> int:
        return len(self.strings)

    def __getitem__(self, index) -> "WordpieceBatch":
        if isinstance(index, int):
            slice_ = slice(index, index + 1)
        else:
            slice_ = index
        return WordpieceBatch(
            strings=self.strings[slice_],
            input_ids=self.input_ids[slice_],
            attention_mask=self.attention_mask[slice_],
            lengths=self.lengths[slice_],
            token_type_ids=(
                self.token_type_ids[slice_]
                if self.token_type_ids is not None
                else None
            ),
        )

    @classmethod
    def zeros(cls, lengths: List[int]) -> "WordpieceBatch":
        # TODO
        ...

    @classmethod
    def from_batch_encoding(cls, token_data: BatchEncoding) -> "WordpieceBatch":
        assert (
            isinstance(token_data, BatchEncoding)
            or isinstance(token_data, dict)
        )
        pad_token = token_data.get("pad_token", "[PAD]")
        lengths = [
            len([tok for tok in tokens if tok != pad_token])
            for tokens in token_data["input_texts"]
        ]
        n_seq = len(lengths)
        return cls(
            strings=token_data["input_texts"],
            input_ids=torch2xp(token_data["input_ids"]).reshape((n_seq, -1)),
            attention_mask=torch2xp(token_data["attention_mask"]).reshape((n_seq, -1)),
            lengths=lengths,
            token_type_ids=(
                torch2xp(token_data["token_type_ids"]).reshape((n_seq, -1))
                if "token_type_ids" in token_data
                else None
            )
        )


@dataclass
class TransformerData:
    """Transformer tokens and outputs for one Doc object.
    
    The transformer models return tensors that refer to a whole padded batch
    of documents. These tensors are wrapped into the FullTransformerBatch object.
    The FullTransformerBatch then splits out the per-document data, which is
    handled by this class. Instances of this class are` typically assigned to
    the doc._.trf_data extension attribute.

    Attributes
    ----------
    tokens (Dict): A slice of the tokens data produced by the Huggingface
        tokenizer. This may have several fields, including the token IDs, the
        texts, and the attention mask. See the Huggingface BatchEncoding object
        for details.
    tensors (List[FloatsXd]): The activations for the Doc from the transformer.
        Usually the last tensor that is 3-dimensional will be the most important,
        as that will provide the final hidden state. Generally activations that
        are 2-dimensional will be attention weights. Details of this variable
        will differ depending on the underlying transformer model.
    align (Ragged): Alignment from the Doc's tokenization to the wordpieces.
        This is a ragged array, where align.lengths[i] indicates the number of
        wordpiece tokens that token i aligns against. The actual indices are
        provided at align[i].dataXd.
    """

    tokens: WordpieceBatch
    tensors: List[FloatsXd]
    align: Ragged

    @classmethod
    def empty(cls) -> "TransformerData":
        align = Ragged(numpy.zeros((0,), dtype="i"), numpy.zeros((0,), dtype="i"))
        return cls(tokens={}, tensors=[], align=align)

    @classmethod
    def zeros(cls, length: int, width: int, *, xp=numpy) -> "TransformerData":
        """Create a valid TransformerData container for a given shape, filled
        with zeros."""
        return cls(
            tokens=WordpieceBatch.zeros([length]),
            tensors=[xp.zeros((1, length, width), dtype="f")],
            align=Ragged(numpy.arange(length), numpy.ones((length,), dtype="i")),
        )

    @property
    def width(self) -> int:
        for tensor in reversed(self.tensors):
            if len(tensor.shape) == 3:
                return tensor.shape[-1]
        else:
            raise ValueError("Cannot find last hidden layer")


@dataclass
class FullTransformerBatch:
    """Holds a batch of input and output objects for a transformer model. The
    data can then be split to a list of `TransformerData` objects to associate
    the outputs to each `Doc` in the batch.

    Attributes
    ----------
    spans (List[List[Span]]): The batch of input spans. The outer list refers
        to the Doc objects in the batch, and the inner list are the spans for
        that `Doc`. Note that spans are allowed to overlap or exclude tokens,
        but each Span can only refer to one Doc (by definition). This means that
        within a Doc, the regions of the output tensors that correspond to each
        Span may overlap or have gaps, but for each Doc, there is a non-overlapping
        contiguous slice of the outputs.
    tokens (WordpieceBatch): The output of the Huggingface tokenizer.
    tensors (List[torch.Tensor]): The output of the transformer model.
    align (Ragged): Alignment from the spaCy tokenization to the wordpieces.
        This is a ragged array, where align.lengths[i] indicates the number of
        wordpiece tokens that token i aligns against. The actual indices are
        provided at align[i].dataXd.
    """

    spans: List[List[Span]]
    tokens: WordpieceBatch
    tensors: List[torch.Tensor]
    align: Ragged
    cached_doc_data: Optional[List[TransformerData]] = None

    @classmethod
    def empty(cls, nr_docs) -> "FullTransformerBatch":
        spans = [[] for i in range(nr_docs)]
        doc_data = [TransformerData.empty() for i in range(nr_docs)]
        align = Ragged(numpy.zeros((0,), dtype="i"), numpy.zeros((0,), dtype="i"))
        return cls(
            spans=spans, tokens={}, tensors=[], align=align, cached_doc_data=doc_data
        )

    @property
    def doc_data(self) -> List[TransformerData]:
        """The outputs, split per spaCy Doc object."""
        if self.cached_doc_data is None:
            self.cached_doc_data = self.split_by_doc()
        return self.cached_doc_data

    def unsplit_by_doc(self, arrays: List[List[Floats3d]]) -> "FullTransformerBatch":
        """Return a new FullTransformerBatch from a split batch of activations,
        using the current object's spans, tokens and alignment.

        This is used during the backward pass, in order to construct the gradients
        to pass back into the transformer model.
        """
        xp = get_array_module(arrays[0][0])
        return FullTransformerBatch(
            spans=self.spans,
            tokens=self.tokens,
            tensors=[xp2torch(xp.vstack(x)) for x in transpose_list(arrays)],
            align=self.align,
        )

    def split_by_doc(self) -> List[TransformerData]:
        """Split a TransformerData that represents a batch into a list with
        one TransformerData per Doc.
        """
        flat_spans = []
        for doc_spans in self.spans:
            flat_spans.extend(doc_spans)
        token_positions = get_token_positions(flat_spans)
        outputs = []
        start = 0
        prev_tokens = 0
        for doc_spans in self.spans:
            if len(doc_spans) == 0 or len(doc_spans[0]) == 0:
                outputs.append(TransformerData.empty())
                token_count = 0
            else:
                start_i = token_positions[doc_spans[0][0]]
                end_i = token_positions[doc_spans[-1][-1]] + 1
                end = start + len(doc_spans)
                doc_tokens = self.tokens[start:end]
                doc_tensors = [torch2xp(t[start:end]) for t in self.tensors]
                doc_align = self.align[start_i:end_i]
                doc_align.data = doc_align.data - prev_tokens
                outputs.append(
                    TransformerData(
                        tokens=doc_tokens,
                        tensors=doc_tensors,  # type: ignore
                        align=doc_align,
                    )
                )
                token_count = sum(doc_tokens.lengths)
            prev_tokens += token_count
            start += len(doc_spans)
        return outputs
