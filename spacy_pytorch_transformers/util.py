from typing import Union, List, Sequence, Callable, Any, Optional
from dataclasses import dataclass
import pytorch_transformers as pytt
from thinc.neural.ops import get_array_module
from thinc.extra.wrappers import torch2xp
import numpy

from . import _tokenizers


Array = Union["numpy.ndarray", "cupy.ndarray"]
Optimizer = Callable[[Array, Array, Optional[int]], None]
Dropout = Optional[float]


SPECIAL_TOKENS: Sequence[str] = (
    "[CLS]",
    "[BOS]",
    "[SEP]",
    "<cls>",
    "<sep>",
    "<|endoftext|>",
    "<s>",
    "</s>",
)


@dataclass
class Activations:
    lh: Array
    po: List[Array]
    ah: List[Array]
    aa: List[Array]
    is_grad: bool = False

    @classmethod
    def blank(cls, is_grad=False):
        return cls(numpy.zeros((0, 0), dtype="f"), [], [], [], is_grad=is_grad)

    @classmethod
    def from_pytt(cls, fields, *, is_grad=False) -> "Activations":
        """Create Activations from the output tuples produced by PyTorch Transformers.
        Includes converting torch tensors to xp, and handling missing values.
        """
        fields = list(fields)
        # Make sure we have 4 elements
        while len(fields) < 4:
            fields.append([])
        # Normalize None to []
        fields = [f if f is not None else f for f in fields]
        # lh: last hidden
        # po: pooler_output
        # ah: all_hidden
        # aa: all_attention
        lh, po, ah, aa = fields
        # Convert last_hidden_state to xp
        lh = torch2xp(lh)
        # Normalize "None" value for pooler output
        if isinstance(po, tuple) and all(x is None for x in po):
            po = []
        po = list(map(torch2xp, po))
        ah = list(map(torch2xp, ah))
        aa = list(map(torch2xp, aa))
        return cls(lh, po, ah, aa, is_grad=is_grad)

    @classmethod
    def join(cls, sub_acts: List["Activations"]) -> "Activations":
        """Concatenate activations from subsequences."""
        xp = get_array_module(sub_acts[0].lh)
        lh: Array = xp.vstack([x.lh for x in sub_acts])
        return cls(lh, [], [], [], is_grad=sub_acts[0].is_grad)

    def __len__(self) -> int:
        return len(self.lh)

    def get_slice(self, x, y) -> "Activations":
        lh = self.lh[x, y]
        # TODO: Support other output fields
        return Activations(lh, [], [], [], is_grad=self.is_grad)

    def split(self, ops: Any, lengths: List[int]) -> List["Activations"]:
        """Split into a list of Activation objects."""
        last_hiddens = ops.unflatten(self.lh, lengths)
        # TODO: Support other output fields
        return [
            Activations(lh, [], [], [], is_grad=self.is_grad) for lh in last_hiddens
        ]

    @property
    def has_lh(self) -> bool:
        return bool(self.lh.size)

    @property
    def has_po(self) -> bool:
        return bool(sum(len(x) for x in self.po))

    @property
    def has_ah(self) -> bool:
        return bool(sum(len(x) for x in self.ah))

    @property
    def has_aa(self) -> bool:
        return bool(sum(len(x) for x in self.aa))


def get_pytt_config(name):
    """Map a name to the appropriate pytorch_transformers.*Config class."""
    name = name.lower()
    if "bert" in name:
        return pytt.BertConfig
    elif "xlnet" in name:
        return pytt.XLNetConfig
    elif "gpt2" in name:
        return pytt.GPT2Config
    elif "xlm" in name:
        return pytt.XLMConfig
    else:
        raise ValueError(f"Unsupported PyTT config name: '{name}'")


def get_pytt_model(name):
    """Map a name to the appropriate pytorch_transformers.*Model class."""
    name = name.lower()
    if "bert" in name:
        return pytt.BertModel
    elif "xlnet" in name:
        return pytt.XLNetModel
    elif "gpt2" in name:
        return pytt.GPT2Model
    elif "xlm" in name:
        return pytt.XLMModel
    else:
        raise ValueError(f"Unsupported PyTT config name: '{name}'")


def get_pytt_tokenizer(name):
    """Get a pytorch_transformers.*Tokenizer class from a name."""
    name = name.lower()
    if "bert" in name:
        return _tokenizers.SerializableBertTokenizer
    elif "xlnet" in name:
        return _tokenizers.SerializableXLNetTokenizer
    elif "gpt2" in name:
        return _tokenizers.SerializableGPT2Tokenizer
    elif "xlm" in name:
        return _tokenizers.SerializableXLMTokenizer
    else:
        raise ValueError(f"Unsupported PyTT config name: '{name}'")


def pad_batch(batch: List[Array], xp=numpy) -> Array:
    """Pad a batch with zeros so that sequences are the same length, and form
    them into a single array.
    """
    max_len = max((len(seq) or 0) for seq in batch)
    padded: List[Array] = []
    seq: Array
    for seq in batch:
        # Ugh, numpy.pad sucks.
        if isinstance(seq, list):
            pad_desc = [[0, 0]]
        else:
            pad_desc = [[0, 0] for _ in seq.shape]
        pad_desc[0][1] = max_len - len(seq)
        padded.append(xp.pad(seq, pad_desc, mode="constant", constant_values=(0, 0)))
    return xp.vstack(padded)


def pad_batch_activations(batch: List[Activations]) -> Activations:
    xp = get_array_module(batch[0])
    lh = pad_batch([x.lh for x in batch], xp=xp)
    lh = lh.reshape((len(batch), -1, lh.shape[-1]))
    return Activations(lh, [], [], [], is_grad=batch[0].is_grad)


def batch_by_length(
    seqs: Union[List[Array], List[Activations]], max_words: int
) -> List[List[int]]:
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order. Batches
    may be at most max_words in size, defined as max sequence length * size.
    """
    # Use negative index so we can get sort by position ascending.
    lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort()
    batches: List[List[int]] = []
    batch: List[int] = []
    for length, i in lengths_indices:
        # i = -neg_i
        if not batch:
            batch.append(i)
        elif length * (len(batch) + 1) <= max_words:
            batch.append(i)
        else:
            batches.append(batch)
            batch = [i]
    if batch:
        batches.append(batch)
    # Check lengths match
    assert sum(len(b) for b in batches) == len(seqs)
    # Check no duplicates
    seen = set()
    for b in batches:
        seen.update(b)
    assert len(seen) == len(seqs)
    batches = [list(sorted(batch)) for batch in batches]
    batches.reverse()
    return batches


def unflatten_list(flat: List[Any], lengths: List[int]) -> List[List[Any]]:
    """Unflatten a list into nested sublists, where each sublist i should have
    length lengths[i]."""
    nested: List[List[Any]] = []
    offset = 0
    for length in lengths:
        nested.append(flat[offset : offset + length])
        offset += length
    return nested


def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    flat = []
    for x in nested:
        flat.extend(x)
    return flat


def is_special_token(text: str) -> bool:
    return text in SPECIAL_TOKENS
