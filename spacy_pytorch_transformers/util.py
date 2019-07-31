import numpy
from dataclasses import dataclass
import pytorch_transformers as pytt
from thinc.neural.ops import get_array_module
from thinc.extra.wrappers import torch2xp
import torch

from . import _tokenizers


SPECIAL_TOKENS = (
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
    last_hidden_state: object
    pooler_output: object
    all_hidden_states: object = None
    all_attentions: object = None
    is_grad: bool = False

    @classmethod
    def from_pytt(cls, fields, *, is_grad=False):
        """Create Activations from the output tuples produced by PyTorch Transformers.
        Includes converting torch tensors to xp, and handling missing values.
        """
        fields = list(fields)
        fields[0] = torch2xp(fields[0])
        if len(fields) >= 2:
            po = fields[1]  # pooler output
            if isinstance(po, tuple) and all(x is None for x in po):
                fields[1] = None
            elif isinstance(po, tuple) and all(isinstance(x, torch.Tensor) for x in po):
                fields[1] = [torch2xp(x) for x in po]
                xp = get_array_module(fields[1][0])
                fields[1] = xp.vstack(fields[1])
            else:
                fields[1] = torch2xp(fields[1])
        else:
            fields.append(None)
        return cls(*fields, is_grad=is_grad)

    @classmethod
    def join(cls, sub_acts, *, is_grad=False):
        """Concatenate activations from subsequences."""
        fields = [None, None, None, None]
        if not sub_acts:
            return cls(*fields, None, is_grad=is_grad)
        if sub_acts[0].has_last_hidden_state:
            xp = get_array_module(sub_acts[0].last_hidden_state)
            fields[0] = xp.vstack([sa.last_hidden_state for sa in sub_acts])
        if sub_acts[0].has_pooler_output:
            xp = get_array_module(sub_acts[0].pooler_output)
            # fields[1] = xp.vstack([sa.pooler_output for sa in sub_acts])
        if sub_acts[0].has_all_hidden_states:
            xp = get_array_module(sub_acts[0].all_hidden_states)
            # fields[2] = xp.vstack([sa.all_hidden_states for sa in sub_acts])
        if sub_acts[0].has_all_attentions:
            xp = get_array_module(sub_acts[0].all_hidden_states)
            # fields[3] = xp.vstack([sa.all_attentions for sa in sub_acts])
        return cls(*fields, is_grad=is_grad)

    def __len__(self):
        return sum(
            (
                self.has_last_hidden_state,
                self.has_pooler_output,
                self.has_all_hidden_states,
                self.has_all_attentions,
            )
        )

    def get_slice(self, x, y):
        output = Activations(None, None, None, None, is_grad=self.is_grad)
        if self.has_last_hidden_state:
            output.last_hidden_state = self.last_hidden_state[x, y]
        if self.has_pooler_output:
            output.pooler_output = self.pooler_output[x, y]
        if self.has_all_hidden_states:
            raise NotImplementedError
        if self.has_all_attentions:
            raise NotImplementedError
        return output

    def split(self, ops, shapes):
        """Split into a list of Activation objects."""
        lh_values = [None] * len(shapes)
        po_values = [None] * len(shapes)
        ah_values = [None] * len(shapes)
        aa_values = [None] * len(shapes)
        lh_shapes, po_shapes, ah_shapes, aa_shapes = zip(*shapes)
        if self.has_last_hidden_state:
            lh_lengths = [shape[0] for shape in lh_shapes]
            lh_values = ops.unflatten(self.last_hidden_state, lh_lengths)
        if self.has_pooler_output:
            po_lengths = [shape[0] for shape in po_shapes]
            # po_values = ops.unflatten(self.pooler_output, po_lengths)
        if self.has_all_hidden_states:
            ah_lengths = [shape[0] for shape in ah_shapes]
            # ah_values = ops.unflatten(self.all_hiddens, ah_lengths)
        if self.has_all_attentions:
            aa_lengths = [shape[0] for shape in aa_shapes]
            # aa_values = ops.unflatten(self.all_attentions, aa_lengths)
        outputs = []
        for lh, po, ah, aa in zip(lh_values, po_values, ah_values, aa_values):
            outputs.append(Activations(lh, po, ah, aa, is_grad=self.is_grad))
        return outputs

    @property
    def shapes(self):
        output = [None, None, None, None]
        if self.has_last_hidden_state:
            output[0] = self.last_hidden_state.shape
        if self.has_pooler_output:
            output[1] = self.pooler_output.shape
        if self.has_all_hidden_states:
            output[2] = self.all_hidden_states.shape
        if self.has_all_attentions:
            output[3] = self.all_attentions.shape
        return output

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
    spacy_tokens = list(spacy_tokens)
    wp_tokens = list(wp_tokens)
    offset = 0
    while wp_tokens and wp_tokens[0] in specials:
        wp_tokens.pop(0)
        offset += 1
    while wp_tokens and wp_tokens[-1] in specials:
        wp_tokens.pop(-1)
    if not spacy_tokens or not wp_tokens:
        return []
    try:
        assert "".join(spacy_tokens).lower() == "".join(wp_tokens).lower()
    except AssertionError:
        print(repr("".join(spacy_tokens).lower()))
        print(repr("".join(wp_tokens).lower()))
        raise
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


def pad_batch(batch):
    """Pad a batch with zeros so that sequences are the same length, and form
    them into a single array. Supports only 1d input arrays."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    xp = get_array_module(batch[0])
    for seq in batch:
        # Ugh, numpy.pad sucks.
        if isinstance(seq, list):
            pad_desc = [[0, 0]]
        else:
            pad_desc = [[0, 0] for _ in seq.shape]
        pad_desc[0][1] = max_len - len(seq)
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


def unflatten_list(flat, lengths):
    """Unflatten a list into nested sublists, where each sublist i should have
    length lengths[i]."""
    nested = []
    offset = 0
    for length in lengths:
        nested.append(flat[offset : offset + length])
        offset += length
    return nested


def flatten_list(nested):
    """Flatten a nested list."""
    flat = []
    for x in nested:
        flat.extend(x)
    return flat


def is_special_token(text):
    return text in SPECIAL_TOKENS
