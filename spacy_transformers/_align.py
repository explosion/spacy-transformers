import numpy
from dataclasses import dataclass
from typing import cast, Dict, List, Tuple, Callable, Set
import tokenizations
from spacy.tokens import Span, Doc, Token
from thinc.api import Ragged, Ops
from thinc.types import Ragged, Floats2d, Ints1d


def apply_alignment(ops: Ops, align: Ragged, X: Floats2d) -> Tuple[Ragged, Callable]:
    shape = X.shape
    indices = cast(Ints1d, align.dataXd)
    Y = Ragged(X[indices], cast(Ints1d, ops.asarray(align.lengths)))

    def backprop_apply_alignment(dY: Ragged) -> Floats2d:
        dX = ops.alloc2f(*shape)
        ops.scatter_add(dX, indices, cast(Floats2d, dY.data))
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
