# cython: infer_types=True, boundscheck=False
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from libc.stdint cimport uint32_t, int32_t, int64_t
from libc.stdlib cimport free
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

import numpy


ctypedef unordered_set[uint32_t]* unordered_set_uint32_t_ptr


cpdef get_span2wp_from_offset_mapping(span, wp_char_offsets):
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
        free(s)
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
            if char_idx >= 0 and char_idx < char_to_sp_token_length:
                token_i = char_to_sp_token[char_idx]
            else:
                token_i = -1
            if token_i >= 0 and token_i < alignment_size:
                deref(alignment.at(token_i)).insert(wp_j)
            char_idx += 1
        wp_j += 1
