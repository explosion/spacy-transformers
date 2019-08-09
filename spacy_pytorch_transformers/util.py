from typing import Union, List, Sequence, Callable, Any, Optional
import pytorch_transformers as pytt
import numpy

from . import _tokenizers

try:
    # This allows us to use cupy with mypy, for type checking
    import cupy # noqa
except ImportError:
    pass

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


def pad_batch(
    batch: List[Array], *, axis: int = 0, xp=numpy, to: int = 0, value: int = -1
) -> Array:
    """Pad a batch of arrays with zeros so that sequences are the same
    length, and form them into a single array.
    """
    if not batch:
        return xp.zeros((0, to))

    if isinstance(batch[0], list):
        batch = [xp.array(x, dtype=numpy.int_) for x in batch]

    max_len = max((seq.shape[axis] for seq in batch), default=0)
    if to < 1:
        to = max_len
    elif max_len > to:
        raise ValueError(f"Cannot pad_batch with max len {max_len} to {to}.")

    if isinstance(batch[0], list) or len(batch[0].shape) == 1:
        return _pad_batch_1d(batch, xp=xp, to=to, value=value)
    else:
        return _pad_batch_nd(batch, axis=axis, xp=xp, to=to, value=value)


def _pad_batch_1d(batch: List[Array], *, xp=numpy, to: int, value) -> Array:
    """Pad a batch of lists or 1d arrays with zeros so that sequences are the same
    length, and form them into a single array.
    """
    padded: List[Array] = []
    seq: Array
    values = (0, value)
    pad_desc = [[0, 0]]
    for seq in batch:
        pad_desc[0][1] = to - len(seq)
        padded.append(xp.pad(seq, pad_desc, mode="constant", constant_values=values))
    output = xp.vstack(padded)
    assert output.shape == (len(batch), to), output.shape
    return output


def _pad_batch_nd(
    batch: List[Array], axis: int, *, xp=numpy, to: int = 0, value=-1
) -> Array:
    padded: List[Array] = []
    seq: Array
    values = (0, value)
    pad_desc = [[0, 0] for _ in batch[0].shape]
    for seq in batch:
        # Ugh, numpy.pad sucks.
        pad_desc[axis][1] = to - seq.shape[axis]
        arr = xp.pad(seq, pad_desc, mode="constant", constant_values=values)
        if len(arr.shape) == 2:
            # This prevents us concatenating on the sequence dimension, when what
            # we want is to have a new batch dimension.
            arr = arr.reshape((1, arr.shape[0], arr.shape[1]))
        padded.append(arr)
    output = xp.vstack(padded)
    return output


def batch_by_length(seqs: Union[List[Array]], max_words: int) -> List[List[int]]:
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


def ensure3d(arr: Array, *, axis: int = 1) -> Array:
    """Make sure an array is 3d, inserting a dimension at axis if not."""
    if arr.size == 0:
        return arr.reshape((0, 0, 0))
    elif len(arr.shape) == 3:
        return arr
    elif len(arr.shape) == 2:
        return arr.reshape((arr.shape[0], 1, arr.shape[1]))
    else:
        raise ValueError(f"Cannot make array 3d. Shape: {arr.shape}")


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


def lengths2mask(lengths):
    """Get a boolean mask of which entries in a padded batch are valid, given
    a list of lengths."""
    padded = pad_batch([numpy.ones((L,), dtype="i") for L in lengths])
    padded[padded < 0] = 0
    return padded.reshape((-1,)) >= 1


def is_special_token(text: str) -> bool:
    return text in SPECIAL_TOKENS


def warmup_linear_rates(initial_rate, warmup_steps, total_steps):
    """Generate a series, starting from an initial rate, and then with a warmup
    period, and then a linear decline. Used for learning rates.
    """
    step = 0
    while True:
        if step < warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            factor = max(
                0.0, (total_steps - step) / max(1.0, total_steps - warmup_steps)
            )
        yield factor * initial_rate
        step += 1


from .activations import Activations # noqa
