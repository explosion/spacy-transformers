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
    po: Array
    ah: List[Array]
    aa: List[Array]
    is_grad: bool = False

    @classmethod
    def blank(cls, *, xp=numpy, is_grad=False):
        zeros = xp.zeros((0,), dtype="f")
        return cls(zeros, zeros, [], [], is_grad=is_grad)

    @classmethod
    def from_pytt(cls, fields, *, is_grad=False) -> "Activations":
        """Create Activations from the output tuples produced by PyTorch Transformers.
        Includes converting torch tensors to xp, and handling missing values.
        """
        # lh: last hidden
        # po: pooler_output
        # ah: all_hidden
        # aa: all_attention
        if len(fields) != 4:
            lh = fields[0]
            po = tuple()
            ah = []
            aa = []
        else:
            lh, po, ah, aa = fields
        # Convert last_hidden_state to xp
        lh = torch2xp(lh)
        xp = get_array_module(lh)
        # Normalize "None" value for pooler output
        if isinstance(po, tuple):
            po = xp.zeros((0,), dtype=lh.dtype)
        else:
            po = torch2xp(po).reshape((po.shape[0], 1, po.shape[-1]))
        ah = list(map(torch2xp, ah))
        aa = list(map(torch2xp, aa))
        return cls(lh, po, ah, aa, is_grad=is_grad)

    @classmethod
    def join(cls, sub_acts: List["Activations"]) -> "Activations":
        """Concatenate activations from subsequences."""
        xp = get_array_module(sub_acts[0].lh)
        lh: Array = xp.vstack([x.lh for x in sub_acts])
        po: Array = xp.vstack([x.po for x in sub_acts])
        # Transpose the lists, so that the inner list items refer
        # to the subsequences. Then we can vstack those.
        ah = list(map(xp.vstack, zip(*[x.ah for x in sub_acts])))
        #aa = list(map(xp.vstack, zip(*[x.aa for x in sub_acts])))
        aa = []
        return cls(lh, po, ah, aa, is_grad=sub_acts[0].is_grad)

    def __len__(self) -> int:
        return len(self.lh)

    def get_slice(self, x, y) -> "Activations":
        lh = self.lh[x, y]
        po = self.po[x]
        ah = [self.ah[i][x, y] for i in range(len(self.ah))]
        aa = [self.aa[i][x, y] for i in range(len(self.aa))]
        return Activations(lh, po, ah, aa, is_grad=self.is_grad)

    def split(self, ops: Any, lengths: List[int]) -> List["Activations"]:
        """Split into a list of Activation objects."""
        lh = ops.unflatten(self.lh, lengths)
        po = ops.unflatten(self.po, lengths)
        # Transpose the lists, so that the outer list refers to the subsequences
        if self.ah:
            ah = list(zip(*[ops.unflatten(x, lengths) for x in self.ah]))
        else:
            ah = [[] for _ in lengths]
        if self.aa:
            aa = list(zip(*[ops.unflatten(x, lengths) for x in self.aa]))
        else:
            aa = [[] for _ in lengths]
        assert len(lh) == len(po) == len(ah) == len(aa)
        # Make an Activations object for each subsequence.
        all_args = zip(lh, po, ah, aa)
        return [Activations(*args, is_grad=self.is_grad) for args in all_args]

    @property
    def has_lh(self) -> bool:
        return bool(self.lh.size)

    @property
    def has_po(self) -> bool:
        return bool(self.po.size)

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


def pad_batch(batch: List[Array], *, xp=numpy, to: int=0) -> Array:
    """Pad a batch with zeros so that sequences are the same length, and form
    them into a single array.
    """
    max_len = max((len(seq) for seq in batch), default=0)
    if to < 1:
        to = max_len
    elif max_len > to:
        raise ValueError(f"Cannot pad_batch with max len {max_len} to {to}.")
    padded: List[Array] = []
    seq: Array
    for seq in batch:
        # Ugh, numpy.pad sucks.
        if isinstance(seq, list):
            pad_desc = [[0, 0]]
        else:
            pad_desc = [[0, 0] for _ in seq.shape]
        pad_desc[0][1] = to - len(seq)
        padded.append(xp.pad(seq, pad_desc, mode="constant", constant_values=(0, 0)))
    output = xp.vstack(padded)
    return output


def pad_batch_activations(batch: List[Activations], *, to: int=0) -> Activations:
    if not batch:
        return Activations.blank()
    xp = get_array_module(batch[0])
    lh = pad_batch([x.lh for x in batch], xp=xp, to=to)
    if lh.size:
        lh = lh.reshape((len(batch), -1, lh.shape[-1]))
    po = pad_batch([x.po for x in batch], xp=xp)
    if po.size:
        po = po.reshape((-1, 1, po.shape[-1]))
    # Transpose the lists, and then pad_batch the items
    ah = [pad_batch(list(seq), xp=xp, to=to) for seq in zip(*[x.ah for x in batch])]
    aa = [pad_batch(list(seq), xp=xp, to=to) for seq in zip(*[x.aa for x in batch])]
    return Activations(lh, po, ah, aa, is_grad=batch[0].is_grad)


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
