# cython: infer_types=True, boundscheck=False
from typing import cast, Dict, List, Tuple, Callable, Set, Optional
import numpy
from spacy_alignments.tokenizations import get_alignments
from spacy.tokens import Span, Token
from thinc.api import Ops
from thinc.types import Ragged, Floats2d, Ints1d, Ints2d

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from libc.stdint cimport uint32_t, int32_t, int64_t
from libc.stdlib cimport free
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

ctypedef unordered_set[uint32_t]* unordered_set_uint32_t_ptr


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
    seen_docs = set()
    for span in spans:
        if span.doc in seen_docs:
            continue
        seen_docs.add(span.doc)
        for token in span.doc:
            if token not in token_positions:
                token_positions[token] = len(token_positions)
    return token_positions


def get_alignment_via_offset_mapping(
    spans: List[Span],
    offset_mapping: Ints2d,
) -> Ragged:
    if len(spans) != len(offset_mapping):
        raise ValueError("Cannot align batches of different sizes.")
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions = get_token_positions(spans)
    alignment: List[Set[int]] = [set() for _ in range(len(token_positions))]
    wp_start = 0
    for i, span in enumerate(spans):
        span_offset_mapping = offset_mapping[i]
        span2wp = get_span2wp_from_offset_mapping(span, span_offset_mapping)
        for token, wp_js in zip(span, span2wp):
            position = token_positions[token]
            alignment[position].update(wp_start + j for j in wp_js)
        wp_start += span_offset_mapping.shape[0]
    lengths: List[int] = []
    flat: List[int] = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(sorted(a))
    align = Ragged(
        cast(Ints1d, numpy.array(flat, dtype="i")),
        cast(Ints1d, numpy.array(lengths, dtype="i")),
    )
    return align


def get_alignment(
    spans: List[Span],
    wordpieces: List[List[str]],
    special_tokens: Optional[List[str]] = None,
) -> Ragged:
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
    if special_tokens is None:
        special_tokens = []
    # Tokens can occur more than once, and we need the alignment of each token
    # to its place in the concatenated wordpieces array.
    token_positions = get_token_positions(spans)
    alignment: List[Set[int]] = [set() for _ in range(len(token_positions))]
    wp_start = 0
    for i, (span, wp_toks) in enumerate(zip(spans, wordpieces)):
        sp_toks = [token.text for token in span]
        wp_toks_filtered = wp_toks
        # In the case that the special tokens do not appear in the text, filter
        # them out for alignment purposes so that special tokens like "<s>" are
        # not aligned to the character "s" in the text. (If the special tokens
        # appear in the text, it's not possible to distinguish them from the
        # added special tokens, so they may be aligned incorrectly.)
        if not any([special in span.text for special in special_tokens]):
            wp_toks_filtered = [
                tok if tok not in special_tokens else "" for tok in wp_toks
            ]
        span2wp, wp2span = get_alignments(sp_toks, wp_toks_filtered)
        for token, wp_js in zip(span, span2wp):
            position = token_positions[token]
            alignment[position].update(wp_start + j for j in wp_js)
        wp_start += len(wp_toks)
    lengths: List[int] = []
    flat: List[int] = []
    for a in alignment:
        lengths.append(len(a))
        flat.extend(sorted(a))
    align = Ragged(
        cast(Ints1d, numpy.array(flat, dtype="i")),
        cast(Ints1d, numpy.array(lengths, dtype="i")),
    )
    return align


def get_span2wp_from_offset_mapping(span, wp_char_offsets):
    # create a mapping of char indices to spacy token indices
    cdef int span_idx = span[0].idx
    cdef int span_i = span[0].i
    cdef int char_idx, rel_token_i
    # size is +1 so we don't have to check whether the text has a trailing space
    char_to_sp_token = numpy.full((len(span.text) + 1,), -1, dtype="int32")
    for token in span:
        rel_token_i = token.i - span_i
        for char_idx in range(
                token.idx - span_idx,
                token.idx - span_idx + len(token) + 1,
        ):
            char_to_sp_token[char_idx] = rel_token_i

    # align all wordpiece tokens to one or more spacy token indices
    cdef vector[unordered_set_uint32_t_ptr] alignment
    for _ in range(len(span)):
        alignment.push_back(new unordered_set[uint32_t]())
    _get_span2wp_alignment(
        &alignment,
        numpy.ascontiguousarray(char_to_sp_token),
        char_to_sp_token.size,
        numpy.ascontiguousarray(wp_char_offsets, dtype="int64"),
        wp_char_offsets.shape[0],
    )

    # convert the alignment into a list of aligned wordpiece indices per spacy
    # token index (unsorted at this point)
    cdef unordered_set_uint32_t_ptr s
    cdef vector[unordered_set_uint32_t_ptr].iterator it_v = alignment.begin()
    cdef unordered_set[uint32_t].iterator it_s
    result: List[List[int]] = []
    while it_v != alignment.end():
        result.append([])
        s = deref(it_v)
        it_s = s.begin()
        while it_s != s.end():
            result[-1].append(deref(it_s))
            preinc(it_s)
        del s
        preinc(it_v)
    return result


cdef void _get_span2wp_alignment(
        vector[unordered_set_uint32_t_ptr]* alignment,
        int32_t[::1] char_to_sp_token,
        int char_to_sp_token_length,
        int64_t[:, ::1] wp_char_offsets,
        int wp_char_offsets_length,
    ) nogil:
    cdef int char_idx, start_idx, end_idx, token_i
    cdef int wp_j = 0
    cdef int alignment_size = alignment.size()
    while wp_j < wp_char_offsets_length:
        start_idx = wp_char_offsets[wp_j][0]
        end_idx = wp_char_offsets[wp_j][1]
        char_idx = start_idx
        while char_idx < end_idx:
            if 0 <= char_idx < char_to_sp_token_length:
                token_i = char_to_sp_token[char_idx]
            else:
                token_i = -1
            if 0 <= token_i < alignment_size:
                deref(alignment.at(token_i)).insert(wp_j)
            char_idx += 1
        wp_j += 1
