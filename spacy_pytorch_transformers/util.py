from typing import Union, List, Sequence, Callable, Any, Optional
from dataclasses import dataclass
import pytorch_transformers as pytt
from thinc.neural.ops import get_array_module
from thinc.extra.wrappers import torch2xp
import torch
import re
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
alpha_re = re.compile(r"[^A-Za-z]+")


@dataclass
class Activations:
    lh: Array
    po: List[Array]
    ah: List[Array]
    aa: List[Array]
    is_grad: bool = False

    @classmethod
    def blank(cls, is_grad=False):
        return cls(numpy.zeros((0,0), dtype="f"), [], [], [], is_grad=is_grad)

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
        xp = get_array_module(lh)
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
        # if self.has_pooler_output:
        #    po = self.pooler_output[x, y]
        # else:
        # po = None
        # if self.has_all_hidden_states:
        #    raise NotImplementedError
        # if self.has_all_attentions:
        #    raise NotImplementedError
        return Activations(lh, [], [], [], is_grad=self.is_grad)

    def split(self, ops: Any, lengths: List[int]) -> List["Activations"]:
        """Split into a list of Activation objects."""
        last_hiddens = ops.unflatten(self.lh, lengths)
        return [
            Activations(lh, [], [], [], is_grad=self.is_grad)
            for lh in last_hiddens
        ]
        # lh_values = [None] * len(shapes)
        # po_values = [None] * len(shapes)
        # ah_values = [None] * len(shapes)
        # aa_values = [None] * len(shapes)
        # lh_shapes, po_shapes, ah_shapes, aa_shapes = zip(*shapes)
        # if self.has_last_hidden_state:
        #    lh_lengths = [shape[0] for shape in lh_shapes]
        #    lh_values = ops.unflatten(self.last_hidden_state, lh_lengths)
        # if self.has_pooler_output:
        #    po_lengths = [shape[0] for shape in po_shapes]
        #    # po_values = ops.unflatten(self.pooler_output, po_lengths)
        # if self.has_all_hidden_states:
        #    ah_lengths = [shape[0] for shape in ah_shapes]
        #    # ah_values = ops.unflatten(self.all_hiddens, ah_lengths)
        # if self.has_all_attentions:
        #    aa_lengths = [shape[0] for shape in aa_shapes]
        #    # aa_values = ops.unflatten(self.all_attentions, aa_lengths)
        # outputs = []
        # for lh, po, ah, aa in zip(lh_values, po_values, ah_values, aa_values):
        #    outputs.append(Activations(lh, po, ah, aa, is_grad=self.is_grad))
        # return outputs

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


def align_word_pieces(spacy_tokens, wp_tokens, specials=SPECIAL_TOKENS):
    """Align tokens against word-piece tokens. The alignment is returned as a
    list of lists. If alignment[3] == [4, 5, 6], that means that spacy_tokens[3]
    aligns against 3 tokens: wp_tokens[4], wp_tokens[5] and wp_tokens[6].
    All spaCy tokens must align against at least one element of wp_tokens.
    """
    spacy_tokens = list(spacy_tokens)
    wp_tokens = list(wp_tokens)
    offset = 0
    while wp_tokens and wp_tokens[0] in specials:
        wp_tokens.pop(0)
        offset += 1
    while wp_tokens and wp_tokens[-1] in specials:
        wp_tokens.pop(-1)
    if not wp_tokens:
        return [[] for _ in spacy_tokens]
    elif not spacy_tokens:
        return []
    # Check alignment
    if "".join(spacy_tokens).lower() != "".join(wp_tokens).lower():
        # Force alignment
        spacy_tokens = [alpha_re.sub("", t) for t in spacy_tokens]
        wp_tokens = [alpha_re.sub("", t) for t in wp_tokens]
        spacy_string = "".join(spacy_tokens).lower()
        wp_string = "".join(wp_tokens).lower()
        if spacy_string != wp_string:
            print("spaCy:", spacy_string)
            print("WP:", wp_string)
            raise AssertionError((spacy_string, wp_string))
    output = _align(spacy_tokens, wp_tokens, offset)
    return output


def _align(seq1, seq2, offset):
    # Map character positions to tokens
    map1 = _get_char_map(seq1)
    map2 = _get_char_map(seq2)
    # For each token in seq1, get the set of tokens in seq2
    # that share at least one character with that token.
    alignment = [set() for _ in seq1]
    unaligned = set(range(len(seq2)))
    for char_position in range(map1.shape[0]):
        i = map1[char_position]
        j = map2[char_position]
        alignment[i].add(j)
        if j in unaligned:
            unaligned.remove(j)
    # Sort, make list
    output = [sorted(list(s)) for s in alignment]
    # Expand alignment to adjacent unaligned tokens of seq2
    for indices in output:
        if indices:
            while indices[0] >= 1 and indices[0] - 1 in unaligned:
                indices.insert(0, indices[0] - 1)
            last = len(seq2) - 1
            while indices[-1] < last and indices[-1] + 1 in unaligned:
                indices.append(indices[-1] + 1)
    # Add offset
    for indices in output:
        for i in range(len(indices)):
            indices[i] += offset
    return output


def _get_char_map(seq):
    char_map = numpy.zeros((sum(len(token) for token in seq),), dtype="i")
    offset = 0
    for i, token in enumerate(seq):
        for j in range(len(token)):
            char_map[offset + j] = i
        offset += len(token)
    return char_map


def pad_batch(batch: List[Array]) -> Array:
    """Pad a batch with zeros so that sequences are the same length, and form
    them into a single array. Supports only 1d input arrays."""
    max_len = max((len(seq) or 0) for seq in batch)
    padded: List[Array] = []
    xp = get_array_module(batch[0])
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
    lh = pad_batch([x.lh for x in batch])
    return Activations(lh, [], [], [], is_grad=batch[0].is_grad)


def batch_by_length(
    seqs: Union[List[Array], List[Activations]], min_batch: int
) -> List[List[int]]:
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order. Batches
    must be at least min_batch length long.
    """
    lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort(reverse=True)
    batches: List[List[int]] = []
    batch: List[int] = []
    prev_length = None
    for length, i in lengths_indices:
        if not batch or length == prev_length or len(batch) < min_batch:
            batch.append(i)
        else:
            batches.append(batch)
            batch = [i]
        prev_length = length
    if batch:
        if len(batch) >= min_batch or not batches:
            batches.append(batch)
        else:
            batches[-1].extend(batch)
    # Check lengths match
    assert sum(len(b) for b in batches) == len(seqs)
    # Check no duplicates
    seen = set()
    for b in batches:
        seen.update(b)
    assert len(seen) == len(seqs)
    batches = [list(sorted(batch)) for batch in batches]
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
