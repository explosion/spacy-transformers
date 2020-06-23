from typing import Union, List, Sequence, Callable, Any, Optional
import transformers
import numpy
from spacy.tokens import Doc, Span

from . import _tokenizers

try:
    # This allows us to use cupy with mypy, for type checking
    import cupy  # noqa
except ImportError:
    pass

try:  # Python 3.8
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa: F401


pkg_meta = importlib_metadata.metadata(__name__.split(".")[0])


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


class ATTRS(object):
    alignment = "trf_alignment"
    word_pieces = "trf_word_pieces"
    word_pieces_ = "trf_word_pieces_"
    separator = "trf_separator"
    segments = "trf_segments"
    start = "trf_start"
    end = "trf_end"
    last_hidden_state = "trf_last_hidden_state"
    pooler_output = "trf_pooler_output"
    all_hidden_states = "trf_all_hidden_states"
    all_attentions = "trf_all_attentions"
    d_last_hidden_state = "trf_d_last_hidden_state"
    d_pooler_output = "trf_d_pooler_output"
    d_all_hidden_states = "trf_d_all_hidden_states"
    d_all_attentions = "trf_d_all_attentions"


class PIPES(object):
    wordpiecer = "trf_wordpiecer"
    tok2vec = "trf_tok2vec"
    textcat = "trf_textcat"
    ner = "trf_ner"


LANG_FACTORY = "trf"


def get_config(name):
    """Map a name to the appropriate transformers.*Config class."""
    name = get_config_name(name)
    if name.startswith("roberta"):
        return transformers.RobertaConfig
    elif name.startswith("distilbert"):
        return transformers.DistilBertConfig
    elif name.startswith("bert"):
        return transformers.BertConfig
    elif name.startswith("xlnet"):
        return transformers.XLNetConfig
    elif name.startswith("gpt2"):
        return transformers.GPT2Config
    elif name.startswith("xlm"):
        return transformers.XLMConfig
    else:
        raise ValueError(f"Unsupported transformers config name: '{name}'")


def get_model(name):
    """Map a name to the appropriate transformers.*Model class."""
    name = get_config_name(name)
    if name.startswith("roberta"):
        return transformers.RobertaModel
    elif name.startswith("distilbert"):
        return transformers.DistilBertModel
    elif name.startswith("bert"):
        return transformers.BertModel
    elif name.startswith("xlnet"):
        return transformers.XLNetModel
    elif name.startswith("gpt2"):
        return transformers.GPT2Model
    elif name.startswith("xlm"):
        return transformers.XLMModel
    else:
        raise ValueError(f"Unsupported transformers config name: '{name}'")


def get_tokenizer(name):
    """Get a transformers.*Tokenizer class from a name."""
    name = get_config_name(name)
    if name.startswith("roberta"):
        return _tokenizers.SerializableRobertaTokenizer
    elif name.startswith("distilbert"):
        return _tokenizers.SerializableDistilBertTokenizer
    elif name.startswith("bert"):
        return _tokenizers.SerializableBertTokenizer
    elif name.startswith("xlnet"):
        return _tokenizers.SerializableXLNetTokenizer
    elif name.startswith("gpt2"):
        return _tokenizers.SerializableGPT2Tokenizer
    elif name.startswith("xlm"):
        return _tokenizers.SerializableXLMTokenizer
    else:
        raise ValueError(f"Unsupported transformers config name: '{name}'")


def get_config_name(name):
    try:
        name = transformers.AutoConfig.from_pretrained(name).model_type
    except EnvironmentError:
        name = name.lower()
        name = name.split("/")[-1]
    return name


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


def is_class_token(text: str) -> bool:
    return text == "[CLS]" or text == "<cls>"


def get_segment_ids(name: str, *lengths) -> List[int]:
    if len(lengths) == 1:
        length1 = lengths[0]
        length2 = 0
    elif len(lengths) == 2:
        length1, length2 = lengths
    else:
        msg = f"Expected 1 or 2 segments. Got {len(lengths)}"
        raise ValueError(msg)
    if name.startswith("bert"):
        return get_bert_segment_ids(length1, length2)
    elif name.startswith("distilbert"):
        return get_bert_segment_ids(length1, length2)
    elif name.startswith("xlnet"):
        return get_xlnet_segment_ids(length1, length2)
    elif name.startswith("xlm"):
        return get_xlm_segment_ids(length1, length2)
    elif name.startswith("gpt2"):
        return get_gpt2_segment_ids(length1, length2)
    elif name.startswith("roberta"):
        return get_roberta_segment_ids(length1, length2)

    else:
        raise ValueError(f"Unexpected model name: {name}")


def get_bert_segment_ids(length1: int, length2: int) -> List[int]:
    """Get an array of segment IDs in BERT's format, for an input with one or
    two segments (set length2=0 for one segment). The lengths should be just the
    wordpiece lengths, not including the SEP and CLS tokens.

    According to the HF glue_utils.py module, the convention for BERT is:

    (a) For sequence pairs:
        tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids:   0   0  0    0    0     0     0   0   1  1  1  1   1   1

    (b) For single sequences:
        tokens:   [CLS] the dog is hairy . [SEP]
        type_ids:   0   0   0   0  0     0   0
    """
    if length2:
        return [0] * length1 + [0] + [0] + [1] * length2 + [1]
    else:
        return [0] * length1 + [0] + [0]


def get_xlnet_segment_ids(length1: int, length2: int) -> List[int]:
    """Get an array of segment IDs in XLNet's format, for an input with one or
    two segments (set length2=0 for one segment). The lengths should be just the
    wordpiece lengths, not including the SEP and CLS tokens.

    According to the XLNet code classifer_utils.py module, the convention is:

    (a) For sequence pairs:
        tokens:    is this jack ##son ##ville ? <sep> no it is not . <sep> <cls>
        type_ids:   0   0  0    0      0      0  0    1   1  1  1  1   1   2

    (b) For single sequences:
        tokens:   the dog is hairy . <sep> <cls>
        type_ids:   0   0 0   0    0   0   2
    """
    if length2:
        return [0] * length1 + [0] + [1] * length2 + [1, 2]
    else:
        return [0] * length1 + [0, 2]


def get_xlm_segment_ids(length1: int, length2: int) -> List[int]:
    """Get an array of segment IDs in XLNet's format, for an input with one or
    two segments (set length2=0 for one segment). The lengths should be just the
    wordpiece lengths, not including the SEP and CLS tokens.

    According to the HF glue_utils.py script, the convention is:

    (a) For sequence pairs:
        tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids:   0   0  0    0    0     0     0   0   1  1  1  1   1   1

    (b) For single sequences:
        tokens:   [CLS] the dog is hairy . [SEP]
        type_ids:   0   0   0   0  0     0   0
    """
    if length2:
        return [0] * length1 + [0] + [0] + [1] * length2 + [1]
    else:
        [0] * length1 + [0] + [0]


def get_gpt2_segment_ids(length1: int, length2: int) -> List[int]:
    """Get an array of segment IDs in GPT2's format, for an input with one or
    two segments (set length2=0 for one segment). The lengths should be just the
    wordpiece lengths, not including the SEP and CLS tokens.

    I'm really not sure how this should look? We currently require segment
    boundaries, so we're just using the <|endoftext|> markers in their vocab?

    (a) For sequence pairs:
        tokens:   <|eot|> is this jack ##son ##ville ? <|eot|> no it is not . <|eot|>
        type_ids:   0     0  0    0    0     0       0  0      1  1  1  1   1 1

    (b) For single sequences:
        tokens:   <|eot|> the dog is hairy . <|eot|>
        type_ids:   0      0   0   0   0   0 0
    """
    if not length2:
        return [0] + [0] * length1 + [0]
    else:
        return [0] + [0] * length1 + [0] + [1] * length2 + [1]


def get_roberta_segment_ids(length1: int, length2: int) -> List[int]:
    # Roberta doesn't use Segment IDs
    total = 1 + length1 + 1 + (1 + length2 + 1) * bool(length2)
    return [0] * total


def get_sents(doc: Union[Span, Doc]) -> List[Span]:
    if doc.is_sentenced:
        return list(doc.sents)
    else:
        return [doc[:]]


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


def cyclic_triangular_rate(min_lr, max_lr, period):
    it = 1
    while True:
        # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
        cycle = numpy.floor(1 + it / (2 * period))
        x = numpy.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield min_lr + (max_lr - min_lr) * relative
        it += 1


from .activations import Activations  # noqa
