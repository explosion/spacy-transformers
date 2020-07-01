import numpy
from typing import cast, Dict, List, Tuple, Callable, Set
import tokenizations
from spacy.tokens import Span, Token
from thinc.api import Ops
from thinc.types import Ragged, Floats2d, Ints1d


def apply_alignment(ops: Ops, align: Ragged, X: Floats2d) -> Tuple[Ragged, Callable]:
    """Align wordpiece data (X) to match tokens, and provide a callback to
    reverse it.
 
    This function returns a Ragged array, which represents the fact that one
    token may be aligned against multiple wordpieces. It's a nested list,
    concatenated with a lengths array to indicate the nested structure. 

    The alignment is also a Ragged array, where the lengths indicate how many
    wordpieces each token is aligned against. The output ragged therefore has
    the same lengths as the alignment ragged, which means the output data
    also has the same number of data rows as the alignment. The size of the
    lengths array indicates the number of tokens in the batch.

    The actual alignment is a simple indexing operation:

        for i, index in enumerate(align.data):
            Y[i] = X[index]

    Which is vectorized via numpy advanced indexing:
        
        Y = X[align.data]

    The inverse operation, for the backward pass, uses the 'scatter_add' op
    because one wordpiece may be aligned against multiple tokens. So we need:

        for i, index in enumerate(align.data):
            X[index] += Y[i]

    The addition wouldn't occur if we simply did `X[index] = Y`, so we use
    the scatter_add op.
    """
    if not align.lengths.sum():
        return _apply_empty_alignment(ops, align, X)
    shape = X.shape
    indices = cast(Ints1d, align.dataXd)
    Y = Ragged(X[indices], cast(Ints1d, ops.asarray(align.lengths)))

    def backprop_apply_alignment(dY: Ragged) -> Floats2d:
        assert dY.data.shape[0] == indices.shape[0]
        dX = ops.alloc2f(*shape)
        ops.scatter_add(dX, indices, cast(Floats2d, dY.dataXd))
        return dX

    return Y, backprop_apply_alignment


def _apply_empty_alignment(ops, align, X):
    shape = X.shape
    Y = Ragged(
        ops.alloc2f(align.lengths.shape[0], X.shape[1]),
        ops.alloc1i(align.lengths.shape[0]) + 1,
    )

    def backprop_null_alignment(dY: Ragged) -> Floats2d:
        return ops.alloc2f(*shape)

    return Y, backprop_null_alignment


def get_token_positions(spans: List[Span]) -> Dict[Token, int]:
    token_positions: Dict[Token, int] = {}
    for span in spans:
        for token in span.doc:
            if token not in token_positions:
                token_positions[token] = len(token_positions)
    return token_positions


def get_alignment_via_offset_mapping(spans: List[Span], token_data) -> Ragged:
    # This function uses the offset mapping provided by Huggingface. I'm not
    # sure whether there's a bug here but I'm getting weird errors.
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions = get_token_positions(spans)
    alignment: List[Set[int]] = [set() for _ in range(len(token_positions))]
    wp_start = 0
    for i, span in enumerate(spans):
        for j, token in enumerate(span):
            position = token_positions[token]
            for char_idx in range(token.idx, token.idx + len(token)):
                wp_j = token_data.char_to_token(i, char_idx)
                if wp_j is not None:
                    alignment[position].add(wp_start + wp_j)
        wp_start += len(token_data.input_ids[i])
    lengths: List[int] = []
    flat: List[int] = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(sorted(a))
    align = Ragged(numpy.array(flat, dtype="i"), numpy.array(lengths, dtype="i"))
    return align


def get_alignment(spans: List[Span], wordpieces: List[List[str]]) -> Ragged:
    """Compute a ragged alignment array that records, for each unique token in
    `spans`, the corresponding indices in the flattened `wordpieces` array.
    For instance, imagine you have two overlapping spans:
    
        [[I, like, walking], [walking, outdoors]]

    And their wordpieces are:

        [[I, like, walk, ing], [walk, ing, out, doors]]

    We want to align "walking" against [walk, ing, walk, ing], which have
    indices [2, 3, 4, 5] once the nested wordpieces list is flattened.

    The nested alignment list would be:

    [[0], [1], [2, 3, 4, 5], [6, 7]]
      I   like    walking    outdoors

    Which gets flattened into the ragged array:

    [0, 1, 2, 3, 4, 5, 6, 7]
    [1, 1, 4, 2]

    The ragged format allows the aligned data to be computed via:

    tokens = Ragged(wp_tensor[align.data], align.lengths)

    This produces a ragged format, indicating which tokens need to be collapsed
    to make the aligned array. The reduction is deferred for a later step, so
    the user can configure it. The indexing is especially efficient in trivial
    cases like this where the indexing array is completely continuous.
    """
    if len(spans) != len(wordpieces):
        raise ValueError("Cannot align batches of different sizes.")
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions = get_token_positions(spans)
    alignment: List[Set[int]] = [set() for _ in range(len(token_positions))]
    wp_start = 0
    for i, (span, wp_toks) in enumerate(zip(spans, wordpieces)):
        sp_toks = [token.text for token in span]
        span2wp, wp2span = tokenizations.get_alignments(sp_toks, wp_toks)
        for token, wp_js in zip(span, span2wp):
            position = token_positions[token]
            alignment[position].update(wp_start + j for j in wp_js)
        wp_start += len(wp_toks)
    lengths: List[int] = []
    flat: List[int] = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(sorted(a))
    align = Ragged(numpy.array(flat, dtype="i"), numpy.array(lengths, dtype="i"))
    return align
