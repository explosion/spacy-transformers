from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from spacy.tokens import Doc, Span

_OffsetsT = List[Optional[Tuple[int, int]]]
_AlignmentT = List[List[int]]


def align_docs(
    spans: List[Span], span_offsets: List[List[Optional[Tuple[int, int]]]],
) -> List[List[List[Tuple[int, int]]]]:
    """
    For each token in each doc, get a list of (idx1, idx2) tuples that select
    the rows aligned to that token.
    """
    span_alignments = align_spans(spans, span_offsets)
    by_doc = _group_spans_by_doc(spans)
    i = 0  # Keep track of which span we're up to in the flat list.
    doc_alignments: List[List[List[Tuple[int, int]]]] = []
    for doc, doc_spans in by_doc:
        doc_alignment: List[List[Tuple[int, int]]] = [[] for token in doc]
        for span in doc_spans:
            span_alignment = span_alignments[i]
            for token, token_alignment in zip(span, span_alignment):
                doc_alignment[token.i].extend(token_alignment)
            i += 1
        doc_alignments.append(doc_alignment)
    return doc_alignments


def _group_spans_by_doc(spans: List[Span]) -> List[Tuple[Doc, List[Span]]]:
    """Group spans according to the doc object they refer to."""
    doc_map = defaultdict(list)
    id2doc = {}
    for span in spans:
        id2doc[id(span.doc)] = span.doc
        doc_map[id(span.doc)].append(span)
    return [(id2doc[id_], doc_spans) for id_, doc_spans in doc_map.items()]


def align_spans(
    spans: List[Span], wp_offsets: List[_OffsetsT]
) -> List[List[List[Tuple[int, int]]]]:
    output = []
    for i, span in enumerate(spans):
        alignment = align_offsets(_get_token_offsets(span), wp_offsets[i])
        # Make indices refer to the row (for the span) as well.
        output.append([[(i, j) for j in entry] for entry in alignment])
    return output


def _get_token_offsets(span):
    start = span.start_char
    return [(t.idx - start, t.idx - start + len(t)) for t in span]


def align_offsets(offsets1: _OffsetsT, offsets2: _OffsetsT) -> _AlignmentT:
    """Align tokens in two segments based on their character offsets."""
    # Map characters in offsets2 to their tokens
    map2 = _get_char_map(offsets2)
    alignment: _AlignmentT = []
    for entry in offsets1:
        align_i = []
        if entry is not None:
            start, end = entry
            # Iterate over the characters in offsets1
            for j in range(start, end):
                if j in map2:
                    # Add the token from offsets2 that that character is in.
                    align_i.append(map2[j])
        alignment.append(align_i)
    return alignment


def _get_char_map(offsets: _OffsetsT) -> Dict[int, int]:
    char_map = {}
    for i, entry in enumerate(offsets):
        if entry is not None:
            start, end = entry
            for j in range(start, end):
                char_map[j] = i
    return char_map
