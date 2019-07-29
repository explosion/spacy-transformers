import numpy
from dataclasses import dataclass
import pytorch_transformers as pytt
from thinc.neural.ops import get_array_module
from thinc.extra.wrappers import torch2xp, xp2torch

from . import _tokenizers


SPECIAL_TOKENS = ("[CLS]", "[BOS]", "[SEP]", "<cls>", "<sep>")

@dataclass
class Activations:
    last_hidden_state: object
    pooler_output: object
    all_hidden_states: object=None
    all_attentions: object=None
    is_grad: bool=False

    @classmethod
    def from_pytt(cls, fields, *, is_grad=False):
        """Create Activations from the output tuples produced by PyTorch Transformers.
        Includes converting torch tensors to xp, and handling missing values.
        """
        fields = list(fields)
        fields[0] = torch2xp(fields[0])
        fields[1] = torch2xp(fields[1])
        return cls(*fields, is_grad=is_grad)

    def __len__(self):
        return sum((
            self.has_last_hidden_state,
            self.has_pooler_output,
            self.has_all_hidden_states,
            self.has_all_attentions))

    @property
    def has_last_hidden_state(self):
        return self.last_hidden_state is not None

    @property
    def has_pooler_output(self):
        return self.pooler_output is not None

    @property
    def has_all_hidden_states(self):
        return self.all_hidden_states is not None

    @property
    def has_all_attentions(self):
        return self.all_attentions is not None


def get_pytt_config(name):
    """Map a name to the appropriate pytorch_transformers.*Config class."""
    name = name.lower()
    if "bert" in name:
        return pytt.BertConfig
    elif "xlnet" in name:
        return pytt.XLNetConfig
    elif "openai" in name:
        return pytt.OpenAIGPTConfig
    elif "transfoxl" in name:
        return pytt.TransfoXLConfig
    elif "gpt2" in name:
        return pytt.GPT2Config
    elif "xlm" in name:
        return pytt.XLMConfig
    else:
        raise ValueError(f"Unrecognized PyTT config name: {name}")


def get_pytt_model(name):
    """Map a name to the appropriate pytorch_transformers.*Model class."""
    name = name.lower()
    if "bert" in name:
        return pytt.BertModel
    elif "xlnet" in name:
        return pytt.XLNetModel
    elif "openai" in name:
        return pytt.OpenAIGPTModel
    elif "transfoxl" in name:
        return pytt.TransfoXLModel
    elif "gpt2" in name:
        return pytt.GPT2Model
    elif "xlm" in name:
        return pytt.XLMModel
    else:
        raise ValueError(f"Unrecognized PyTT config name: {name}")


def get_pytt_tokenizer(name):
    """Get a pytorch_transformers.*Tokenizer class from a name."""
    name = name.lower()
    if "bert" in name:
        return _tokenizers.SerializableBertTokenizer
    elif "xlnet" in name:
        return _tokenizers.SerializableXLNetTokenizer
    elif "openai" in name:
        return _tokenizers.SerializableOpenAIGPTTokenizer
    elif "transfoxl" in name:
        return _tokenizers.SerializableTransfoXLTokenizer
    elif "gpt2" in name:
        return _tokenizers.SerializableGPT2Tokenizer
    elif "xlm" in name:
        return _tokenizers.SerializableXLMTokenizer
    else:
        raise ValueError(f"Unrecognized PyTT config name: {name}")


def align_word_pieces(spacy_tokens, wp_tokens, specials=SPECIAL_TOKENS):
    """Align tokens against word-piece tokens. The alignment is returned as a
    list of lists. If alignment[3] == [4, 5, 6], that means that spacy_tokens[3]
    aligns against 3 tokens: wp_tokens[4], wp_tokens[5] and wp_tokens[6].
    All spaCy tokens must align against at least one element of wp_tokens.
    """
    offset = 0
    while wp_tokens and wp_tokens[0] in specials:
        wp_tokens.pop(0)
        offset += 1
    while wp_tokens and wp_tokens[-1] in specials:
        wp_tokens.pop(-1)
    if not spacy_tokens or not wp_tokens:
        return []
    wp_tokens = [wpt.replace("##", "", 1) for wpt in wp_tokens]
    # XLNet uses this as the control char?? Wtf.
    wp_tokens = [wpt.replace("\u2581", "", 1) for wpt in wp_tokens]
    assert "".join(spacy_tokens).lower() == "".join(wp_tokens).lower()
    output = _align(spacy_tokens, wp_tokens, offset)
    return output


def _align(seq1, seq2, offset):
    # Map character positions to tokens
    map1 = _get_char_map(seq1)
    map2 = _get_char_map(seq2)
    # For each token in seq1, get the set of tokens in seq2
    # that share at least one character with that token.
    alignment = [set() for _ in seq1]
    for char_position in range(map1.shape[0]):
        i = map1[char_position]
        j = map2[char_position]
        alignment[i].add(offset + j)
    return [sorted(list(s)) for s in alignment]


def _get_char_map(seq):
    char_map = numpy.zeros((sum(len(token) for token in seq),), dtype="i")
    offset = 0
    for i, token in enumerate(seq):
        for j in range(len(token)):
            char_map[offset + j] = i
        offset += len(token)
    return char_map


def pad_batch(batch):
    """Pad a batch with zeros so that sequences are the same length, and form
    them into a single array. Supports only 1d input arrays."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    xp = get_array_module(batch[0])
    for seq in batch:
        # Ugh, numpy.pad sucks.
        pad_desc = (0, max_len - len(seq))
        padded.append(xp.pad(seq, pad_desc, mode="constant", constant_values=(0, 0)))
    return xp.vstack(padded)


def batch_by_length(seqs, min_batch):
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order. Batches
    must be at least min_batch length long.
    """
    lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort(reverse=True)
    batches = []
    batch = []
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
