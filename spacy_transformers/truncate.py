from typing import Tuple
import numpy
from thinc.types import Ragged, Ints1d, Floats2d, Floats3d
from .data_classes import WordpieceBatch


def truncate_oversize_splits(
    token_data: WordpieceBatch,
    align: Ragged,
    seq_lengths: Ints1d,
    max_length: int
) -> Tuple[WordpieceBatch, Ragged]:
    """Drop wordpieces from inputs that are too long. This can happen because 
    the splitter is based on linguistic tokens, and the number of wordpieces
    that each token is split into is unpredictable, so we can end up with splits
    that have more wordpieces than the model's maximum.
    
    To solve this, we calculate a score for each wordpiece in the split,
    and drop the wordpieces with the highest scores. I can think of a few
    scoring schemes we could use:
    
    a) Drop the ends of longest wordpieces. This scoring would be just:
        position_in_token 
    b) Drop the middles of longest wordpieces. The score would be:
        min(length - position_in_token, position_in_token)
    c) Drop all wordpieces from longest tokens. This would be:
        length
    d) Drop wordpieces from the end of the split. This would be:
        position_in_split
    
    The advantage of a) and b) is that they give some representation to each
    token. The advantage of c) is that it leaves a higher % of tokens with intact
    representations. The advantage of d) is that it leaves contiguous chunks of
    wordpieces intact, and drops from the end.

    I find b) most appealing, but it's also the most complicated. Let's just do
    d) for now.
    """
    if token_data["input_ids"].shape[0] < max_length:
        return token_data, align
    mask = _get_truncation_mask_drop_from_end(seq_lengths, align, max_length)
    return _truncate_tokens(token_data, mask), _truncate_align(align, mask)


def _get_truncation_mask_drop_from_end(
    split_lengths: Ints1d,
    align: Ragged,
    max_length: int
) -> numpy.ndarray:
    """Return a two-dimensional boolean mask, indicating whether wordpieces
    are dropped from their sequences.

    Drop wordpieces from the end of the sequence.
    """
    mask = xp.ones(token_data["input_ids"].shape, dtype="b")
    mask[max_length:] = 0
    return mask


def _truncate_tokens(
    tokens: WordpieceBatch,
    mask: numpy.ndarray
) -> WordpieceBatch:
    batch_size = token_data.batch_size 
    n_wp = mask.shape[0]
    n_keep = mask.sum()
    
    strings = []
    i = 0
    for seq in tokens.strings:
        strings.append([])
        for token in seq:
            if mask[i]:
                strings[-1].append(token)
            i += 1

    def filter_ids(data: Ints2d) -> Ints2d:
        data1d = data.reshape((-1,))
        return data1d[mask].reshape((n_seq,, -1))

    def filter_attn(data: Floats3d) -> Floats3d:
        attn = data.reshape((n_wp, n_wp))
        return attn[mask, mask].reshape((n_seq, n_keep, n_keep))

    return WordpieceBatch(
        strings=strings,
        input_ids=filter_ids(tokens.input_ids),
        input_type_ids=filter_ids(tokens.token_type_ids),
        attention_mask=filter_attn(tokens.attention_mask),
        lengths=numpy.array([len(seq) for seq in strings], dtype="i")
    )


def _truncate_alignment(align: Ragged, mask: numpy.ndarray) -> Ragged:
    # We're going to have fewer wordpieces in the new array, so all of our
    # wordpiece indices in the alignment table will be off --- they'll point
    # to the wrong row. So we need to do three things here:
    #
    # 1) Adjust all the indices in align.dataXd to account for the dropped data
    # 2) Remove the dropped indices from the align.dataXd
    # 3) Calculate new align.lengths 
    # 
    # The wordpiece mapping is easily calculated by the cumulative sum of the
    # mask table.
    # Let's say we have [True, False, False, True]. The mapping of the dropped
    # wordpieces doesn't matter, because we can filter it with the mask. So we
    # have [0, 0, 0, 1], i.e the wordpiece that was
    # at 0 is still at 0, and the wordpiece that was at 3 is now at 1. 
    idx_map = mask.cumsum() - 1
    idx_map[~mask] = -1
    # Step 1: Adjust all the indices in align.dataXd.
    new_align = idx_map[align.dataXd]
    # Step 2: Remove the dropped indices
    dropped = new_align < 0
    new_align = new_align[~dropped]
    # Step 3: Calculate new align.lengths
    new_lengths = align.lengths.copy()
    for i in range(len(align.lengths)):
        slice_ = align[i].data
        drops = dropped[slice_]
        new_lengths[i] -= drops.sum()
    return Ragged(new_align, new_lengths)
