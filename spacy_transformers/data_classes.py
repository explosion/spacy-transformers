from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
import numpy
from transformers.tokenization_utils import BatchEncoding
from transformers.file_utils import ModelOutput
from thinc.types import Ragged, Floats2d, Floats3d, FloatsXd, Ints2d
from thinc.api import NumpyOps, get_array_module, xp2torch, torch2xp
from spacy.tokens import Span
import srsly

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
    attention_mask: Floats2d
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
                self.token_type_ids[slice_] if self.token_type_ids is not None else None
            ),
        )

    def to_hf_dict(self) -> Dict:
        """Return a dict similar to the format produced by the Huggingface
        tokenizer, converting arrays to pytorch tensors as well.
        """
        output = {
            "input_ids": xp2torch(self.input_ids),
            "attention_mask": xp2torch(self.attention_mask),
            "input_texts": self.strings,
        }
        if self.token_type_ids is not None:
            output["token_type_ids"] = xp2torch(self.token_type_ids)
        return output

    @classmethod
    def empty(cls, *, xp=numpy) -> "WordpieceBatch":
        return cls(
            strings=[],
            input_ids=xp.zeros((0, 0), dtype="i"),
            attention_mask=xp.ones((0, 0), dtype="bool"),
            lengths=[],
            token_type_ids=None,
        )

    @classmethod
    def zeros(cls, lengths: List[int], xp=numpy) -> "WordpieceBatch":
        return cls(
            strings=[[""] * length for length in lengths],
            input_ids=xp.array([[0] * length for length in lengths], dtype="i"),
            attention_mask=xp.ones((len(lengths), max(lengths)), dtype="bool"),
            lengths=lengths,
            token_type_ids=None,
        )

    @classmethod
    def from_batch_encoding(cls, token_data: BatchEncoding) -> "WordpieceBatch":
        assert isinstance(token_data, BatchEncoding) or isinstance(token_data, dict)
        pad_token = token_data.get("pad_token", "[PAD]")
        lengths = [
            len([tok for tok in tokens if tok != pad_token])
            for tokens in token_data["input_texts"]
        ]

        numpy_ops = NumpyOps()

        return cls(
            strings=token_data["input_texts"],
            input_ids=numpy_ops.asarray2i(token_data["input_ids"]),
            attention_mask=numpy_ops.asarray2f(token_data["attention_mask"]),
            lengths=lengths,
            token_type_ids=(
                numpy_ops.asarray2i(token_data["token_type_ids"])
                if "token_type_ids" in token_data
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strings": self.strings,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "lengths": self.lengths,
            "token_type_ids": self.token_type_ids,
        }

    def from_dict(self, msg: Dict[str, Any]) -> "WordpieceBatch":
        self.strings = msg["strings"]
        self.input_ids = msg["input_ids"]
        self.attention_mask = msg["attention_mask"]
        self.lengths = msg["lengths"]
        self.token_type_ids = msg["token_type_ids"]
        return self


@dataclass
class TransformerData:
    """Transformer tokens and outputs for one Doc object.

    The transformer models return tensors that refer to a whole padded batch
    of documents. These tensors are wrapped into the FullTransformerBatch object.
    The FullTransformerBatch then splits out the per-document data, which is
    handled by this class. Instances of this class are typically assigned to
    the doc._.trf_data extension attribute.

    Attributes
    ----------
    wordpieces (WordpieceBatch): A slice of the wordpiece token data produced
        by the Huggingface tokenizer.
    model_output (ModelOutput): The model output from the transformer model,
        determined by the model and transformer config.
    align (Ragged): Alignment from the Doc's tokenization to the wordpieces.
        This is a ragged array, where align.lengths[i] indicates the number of
        wordpiece tokens that token i aligns against. The actual indices are
        provided at align[i].dataXd.
    """

    wordpieces: WordpieceBatch
    model_output: ModelOutput
    align: Ragged

    @classmethod
    def empty(cls) -> "TransformerData":
        align = Ragged(numpy.zeros((0,), dtype="i"), numpy.zeros((0,), dtype="i"))
        return cls(
            wordpieces=WordpieceBatch.empty(), model_output=ModelOutput(), align=align
        )

    @classmethod
    def zeros(cls, length: int, width: int, *, xp=numpy) -> "TransformerData":
        """Create a valid TransformerData container for a given shape, filled
        with zeros."""
        return cls(
            wordpieces=WordpieceBatch.zeros([length], xp=xp),
            model_output=ModelOutput(
                last_hidden_state=xp.zeros((1, length, width), dtype="f")
            ),
            align=Ragged(numpy.arange(length), numpy.ones((length,), dtype="i")),
        )

    @property
    def tensors(self) -> Tuple[Union[FloatsXd, List[FloatsXd]]]:
        return self.model_output.to_tuple()

    @property
    def tokens(self) -> Dict[str, Any]:
        """Deprecated. A dict with the wordpiece token data."""
        return self.wordpieces.to_hf_dict()

    @property
    def width(self) -> int:
        if "last_hidden_state" in self.model_output:
            return self.model_output.last_hidden_state.shape[-1]
        else:
            raise ValueError("Cannot find last hidden state")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wordpieces": self.wordpieces.to_dict(),
            "model_output": self.model_output,
            "align": [self.align.dataXd, self.align.lengths],
        }

    def from_dict(self, msg: Dict[str, Any]) -> "TransformerData":
        self.wordpieces = WordpieceBatch.empty().from_dict(msg["wordpieces"])
        self.model_output = ModelOutput(msg["model_output"])
        self.align = Ragged(*msg["align"])
        return self

    def to_bytes(self) -> bytes:
        return srsly.msgpack_dumps(self.to_dict())

    def from_bytes(self, byte_string: bytes) -> "TransformerData":
        msg = srsly.msgpack_loads(byte_string)
        self.from_dict(msg)
        return self


@srsly.msgpack_encoders("transformerdata")
def serialize_transformer_data(obj, chain=None):
    if isinstance(obj, TransformerData):
        return {"__transformerdata__": obj.to_dict()}
    return obj if chain is None else chain(obj)


@srsly.msgpack_decoders("transformerdata")
def deserialize_transformer_data(obj, chain=None):
    if "__transformerdata__" in obj:
        return TransformerData.empty().from_dict(obj["__transformerdata__"])
    return obj if chain is None else chain(obj)


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
    wordpieces (WordpieceBatch): Token data from the Huggingface tokenizer.
    model_output (ModelOutput): The output of the transformer model.
    align (Ragged): Alignment from the spaCy tokenization to the wordpieces.
        This is a ragged array, where align.lengths[i] indicates the number of
        wordpiece tokens that token i aligns against. The actual indices are
        provided at align[i].dataXd.
    """

    spans: List[List[Span]]
    wordpieces: WordpieceBatch
    model_output: ModelOutput
    align: Ragged
    cached_doc_data: Optional[List[TransformerData]] = None

    @classmethod
    def empty(cls, nr_docs) -> "FullTransformerBatch":
        spans = [[] for _ in range(nr_docs)]
        doc_data = [TransformerData.empty() for _ in range(nr_docs)]
        align = Ragged(numpy.zeros((0,), dtype="i"), numpy.zeros((0,), dtype="i"))
        return cls(
            spans=spans,
            wordpieces=WordpieceBatch.empty(),
            model_output=ModelOutput(),
            align=align,
            cached_doc_data=doc_data,
        )

    @property
    def tensors(self) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        return self.model_output.to_tuple()

    @property
    def tokens(self) -> Dict[str, Any]:
        """Deprecated. Dict formatted version of the self.wordpieces data,
        with values converted to PyTorch tensors.
        """
        return self.wordpieces.to_hf_dict()

    @property
    def doc_data(self) -> List[TransformerData]:
        """The outputs, split per spaCy Doc object."""
        if self.cached_doc_data is None:
            self.cached_doc_data = self.split_by_doc()
        return self.cached_doc_data

    def unsplit_by_doc(self, arrays: List[List[Floats3d]]) -> "FullTransformerBatch":
        """Return a new FullTransformerBatch from a split batch of activations,
        using the current object's spans, wordpieces and alignment.

        This is used during the backward pass, in order to construct the gradients
        to pass back into the transformer model.
        """
        xp = get_array_module(arrays[0][0])
        # construct a dummy ModelOutput with the tensor values
        model_output = ModelOutput()
        for i, x in enumerate(transpose_list(arrays)):
            model_output[f"output_{i}"] = xp2torch(xp.vstack(x))
        return FullTransformerBatch(
            spans=self.spans,
            wordpieces=self.wordpieces,
            model_output=model_output,
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
                continue
            start_i = token_positions[doc_spans[0][0]]
            end_i = token_positions[doc_spans[-1][-1]] + 1
            end = start + len(doc_spans)
            doc_tokens = self.wordpieces[start:end]
            doc_align = self.align[start_i:end_i]
            doc_align.data = doc_align.data - prev_tokens
            model_output = ModelOutput()
            last_hidden_state = self.model_output.last_hidden_state
            for key, output in self.model_output.items():
                if isinstance(output, torch.Tensor):
                    model_output[key] = torch2xp(output[start:end])
                elif (
                    isinstance(output, tuple)
                    and all(isinstance(t, torch.Tensor) for t in output)
                    and all(t.shape[0] == last_hidden_state.shape[0] for t in output)
                ):
                    model_output[key] = [torch2xp(t[start:end]) for t in output]
            outputs.append(
                TransformerData(
                    wordpieces=doc_tokens,
                    model_output=model_output,
                    align=doc_align,
                )
            )
            prev_tokens += doc_tokens.input_ids.size
            start += len(doc_spans)
        return outputs
