import numpy
from dataclasses import dataclass
from typing import cast, Dict, List, Tuple, Callable
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


def get_alignment(spans: List[Span], wordpieces: List[List[str]]) -> Ragged:
    if len(spans) != len(wordpieces):
        raise ValueError("Cannot align batches of different sizes.")
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions: Dict[Tuple[int, int], int] = {}
    for span in spans:
        for token in span:
            key = (id(span.doc), token.idx)
            if key not in token_positions:
                token_positions[key] = len(token_positions) 
    alignment: List[List[int]] = [[] for _ in range(len(token_positions))]
    sp_start = 0
    wp_start = 0
    for i, (span, wp_toks) in enumerate(zip(spans, wordpieces)):
        sp_toks = [token.text for token in span]
        span2wp, wp2span = tokenizations.get_alignments(sp_toks, wp_toks)
        for token, wp_js in zip(span, span2wp):
            key = (id(span.doc), token.idx)
            position = token_positions[key]
            alignment[position].extend([wp_start + j for j in wp_js])
        wp_start += len(wp_toks)
    lengths = []
    flat = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(a)
    return Ragged(numpy.array(flat, dtype="i"), numpy.array(lengths, dtype="i"))


def slice_alignment(alignment: Ragged, start: int, end: int, sp_lengths, wp_lengths) -> Ragged:
    """Extract the alignment for a subset of the batch, adjusting the
    indices accordingly."""
    sp_start = sum(sp_lengths[:start])
    sp_end = sum(sp_lengths[:end])
    wp_start = sum(wp_lengths[:start])
    wp_end = sum(wp_lengths[:end])
    # Get the slice, and adjust the data so they point to the right part
    # of the target array. We're trusting that nothing will point past the
    # ends.
    tok2trf = alignment[sp_start:sp_end]
    tok2trf.data -= wp_start
    return tok2trf
