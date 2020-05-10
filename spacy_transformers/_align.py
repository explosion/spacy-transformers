from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import tokenizations
from spacy.tokens import Doc, Span


_OffsetsT = List[Optional[Tuple[int, int]]]
_AlignmentT = List[List[int]]


class BatchAlignment:
    """Alignment for a batch of texts between wordpieces and tokens."""
    wp2tok: Ints1d
    tok2wp: Ints1d
    wp_lengths: List[int]
    tok_lengths: List[int]

    @classmethod
    def from_strings(cls, wp: List[List[str]], tok: List[List[str]]):
        wp2tok, tok2wp = _align_batch(wp, tok)
        wp_lengths = [len(x) for x in wp]
        tok_lengths = [len(x) for x in tok]
        return cls(wp2tok, tok2wp, wp_lengths, tok_lengths)

    def slice(self, start: int, end: int) -> Tuple[Ints1d, Ints1d]:
        """Extract the alignment for a subset of the batch, adjusting the
        indices accordingly."""
        wp_start = sum(self.wp_lengths[:start])
        wp_end = sum(self.wp_lengths[:end])
        tok_start = sum(self.tok_lengths[:start])
        tok_end = sum(self.tok_lengths[:end])
        # Get the slice, and adjust the data so they point to the right part
        # of the target array. We're trusting that nothing will point past the
        # ends.
        wp2tok = self.wp2tok[wp_start : wp_end] - tok_start
        tok2wp = self.tok2wp[tok_start : tok_end] - wp_start
        # Handle, which indicates null alignment
        wp2tok[wp2tok < 0] = -1
        tok2wp[tok2wp < 0] = -1
        return wp2tok, tok2wp



def _align_batch(A: List[List[str]], B: List[List[str]]) -> Tuple[Ints1d, Ints1d]:
    if len(A) != len(B):
        raise ValueError("Cannot align batches of different sizes.")
    batch_size = len(A)
    a_stride = max(len(a) for a in A, default=0)
    b_stride = max(len(b) for b in B), default=0
    A2B = numpy.zeros((batch_size * a_stride,), dtype="i")
    B2A = numpy.zeros((batch_size * b_stride,), dtype="i")
    A2B.fill(-1)
    B2A.fill(-1)
    for i, (a, b) in enumerate(zip(A, B)):
        a2b, b2a = tokenizations.get_alignments(a, b)
        for a_j, b_js in enumerate(a2b):
            for b_j in b_js:
                B2A[i * b_stride + b_j] = i * a_stride + a_j
        for b_j, a_js in enumerate(a2b):
            for a_j in a_js:
                A2B[i * a_stride + a_j] = i * b_stride + b_j
    # To use the alignment table, we do like:
    # 
    # A_ = ops.scatter_add(A, B2A, B)
    # B_ = ops.scatter_add(B, A2B, A)
    # 
    # This will iterate over the table and do A[B2A[i]] += B[i] etc
    # Note that the indices are flat, so you have to reshape your table to 2d.
    # You also have to append a dummy value to the end of the destination array
    # and strip it, so that there's a way to do non-alignment.
    return A2B, B2A
