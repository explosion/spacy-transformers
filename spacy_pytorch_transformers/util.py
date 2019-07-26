import spacy.gold
from thinc.neural.ops import get_array_module


def align_word_pieces(spacy_tokens, wp_tokens, specials=("[CLS]", "[BOS]", "[SEP]")):
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
    assert "".join(spacy_tokens).lower() == "".join(wp_tokens).lower()
    if len(wp_tokens) > len(spacy_tokens):
        cost, w2s, s2w, multi_w2s, multi_s2w = spacy.gold.align(wp_tokens, spacy_tokens)
    else:
        cost, s2w, w2s, multi_s2w, multi_w2s = spacy.gold.align(spacy_tokens, wp_tokens)
    output = []
    for i in range(len(spacy_tokens)):
        if s2w[i] != -1:
            output.append([offset + s2w[i]])
        else:
            output.append([])
    for i, j in multi_w2s.items():
        output[j].append(offset + i)
    return output


def pad_batch(batch, value=0):
    """Pad a batch so that sequences are the same length, and form them into
    a single array."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    xp = get_array_module(batch[0])
    for seq in batch:
        # Ugh, numpy.pad sucks.
        pad_desc = (0, max_len - len(seq))
        padded.append(
            xp.pad(seq, pad_desc, mode="constant", constant_values=(0, value))
        )
    return xp.vstack(padded)


def batch_by_length(seqs, min_batch, min_density):
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order. Two
    constraints are available for the batching:

    * min_batch: Try to form batches of at least N members.
    * min_density: Try to form batches where ratio of actual to padding elements
        is at least N.

    The min_density constraint has priority if there's a conflict.
    """
    lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort(reverse=True)
    batches = []
    batch = []
    prev_length = None
    for length, i in lengths_indices:
        if not batch or length == prev_length:
            batch.append(i)
        elif len(batch) >= min_batch:
            batches.append(batch)
            batch = [i]
        else:
            # Would adding this to the batch screw up the batch density?
            # If not, go ahead and add it to the batch. Otherwise, make a new
            # batch.
            active = sum(len(seqs[b]) for b in batch) + length
            total = (len(batch) + 1) * len(seqs[batch[0]])
            if (active / total) >= min_density:
                batch.append(i)
            else:
                batches.append(batch)
                batch = [i]
        prev_length = length
    if batch:
        batches.append(batch)
    assert sum(len(b) for b in batches) == len(seqs)
    return batches
