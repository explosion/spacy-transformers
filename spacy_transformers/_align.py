import numpy
from typing import cast, Dict, List, Tuple, Callable, Set
import tokenizations
from spacy.tokens import Span, Token
from thinc.api import Ops
from thinc.types import Ragged, Floats2d, Ints1d


def apply_alignment(ops: Ops, align: Ragged, X: Floats2d) -> Tuple[Ragged, Callable]:
    """Align wordpiece data (X) to match tokens."""
    shape = X.shape
    indices = cast(Ints1d, align.dataXd)
    Y = Ragged(X[indices], cast(Ints1d, ops.asarray(align.lengths)))

    def backprop_apply_alignment(dY: Ragged) -> Floats2d:
        dX = ops.alloc2f(*shape)
        # TODO: We get errors here when we align against no tokens, because
        # the 0 in the lengths means the data is misaligned.
        try:
            ops.scatter_add(dX, indices, dY_)
        except:
            print(shape)
            print(dY.lengths)
            raise
        return dX

    return Y, backprop_apply_alignment


def get_token_positions(spans: List[Span]) -> Dict[Tuple[Token, int], int]:
    token_positions: Dict[Token, int] = {}
    for span in spans:
        for token in span:
            if token not in token_positions:
                token_positions[token] = len(token_positions)
    return token_positions


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
