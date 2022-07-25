# cython: infer_types=True, boundscheck=False
cimport numpy as np
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from libc.stdint cimport uint32_t, int32_t, int64_t
from libc.stdlib cimport free
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

import numpy


ctypedef unordered_set[uint32_t]* unordered_set_uint32_t_ptr


cpdef get_span2wp_from_offset_mapping(span, span_mapping):
    # create a mapping of characters to token indices for the spacy span
    cdef int span_idx = span[0].idx
    cdef int span_i = span[0].i
    cdef int i, rel_token_i
    # size is +1 so we don't have to check whether the text has a trailing space
    char_to_token = numpy.empty((len(span.text) + 1,), dtype="int32")
    char_to_token.fill(-1)
    for token in span:
        rel_token_i = token.i - span_i
        for i in range(token.idx - span_idx, token.idx - span_idx + len(token) + 1):
            char_to_token[i] = rel_token_i

    # align all wordpiece tokens to one or more spacy token indicies
    cdef vector[unordered_set_uint32_t_ptr] alignment
    for i in range(len(span)):
        alignment.push_back(new unordered_set[uint32_t]())
    _get_span2wp_alignment(
        &alignment,
        numpy.ascontiguousarray(span_mapping, dtype="int64"),
        span_mapping.size,
        numpy.ascontiguousarray(char_to_token),
    )

    # convert the alignment into a list of aligned wordpiece indices per spacy
    # token index
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
        int64_t[::1] span_mapping,
        int span_mapping_length,
        int32_t[::1] char_to_token,
    ) nogil:
    cdef uint32_t char_idx, start_idx, end_idx, token_i, wp_j
    wp_j = 0
    # the span_mapping has size len(wordpieces) * 2, index as a flattened
    # array
    while wp_j < span_mapping_length / 2:
        start_idx = span_mapping[wp_j*2]
        end_idx = span_mapping[wp_j*2 + 1]
        char_idx = start_idx
        while char_idx < end_idx:
            token_i = char_to_token[char_idx]
            # TODO: add a warning if not mapped?
            if token_i >= 0:
                deref(alignment.at(token_i)).insert(wp_j)
            char_idx += 1
        wp_j += 1
