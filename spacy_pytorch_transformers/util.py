import numpy
from thinc.neural.ops import get_array_module

SPECIAL_TOKENS = (
    "[CLS]", "[BOS]", "[SEP]", "<cls>", "<sep>"
)

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
        alignment[i].add(offset+j)
    return [sorted(list(s)) for s in alignment]


def _get_char_map(seq):
    char_map = numpy.zeros((sum(len(token) for token in seq),), dtype='i')
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
        padded.append(
            xp.pad(seq, pad_desc, mode="constant", constant_values=(0, 0))
        )
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
