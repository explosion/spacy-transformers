import numpy
from dataclasses import dataclass
from typing import List, Tuple
import tokenizations
from thinc.api import Ragged


@dataclass
class BatchAlignment:
    """Alignment for a batch of texts between wordpieces and tokens."""

    wp2tok: Ragged
    tok2wp: Ragged
    wp_lengths: List[int]
    tok_lengths: List[int]

    @classmethod
    def from_strings(cls, wp: List[List[str]], tok: List[List[str]]):
        wp2tok, tok2wp = _align_batch(wp, tok)
        wp_lengths = [len(x) for x in wp]
        tok_lengths = [len(x) for x in tok]
        return cls(wp2tok, tok2wp, wp_lengths, tok_lengths)

    def slice(self, start: int, end: int) -> Tuple[Ragged, Ragged]:
        """Extract the alignment for a subset of the batch, adjusting the
        indices accordingly."""
        wp_start = sum(self.wp_lengths[:start])
        wp_end = sum(self.wp_lengths[:end])
        tok_start = sum(self.tok_lengths[:start])
        tok_end = sum(self.tok_lengths[:end])
        # Get the slice, and adjust the data so they point to the right part
        # of the target array. We're trusting that nothing will point past the
        # ends.
        wp2tok = self.wp2tok[wp_start:wp_end]
        tok2wp = self.tok2wp[tok_start:tok_end]
        wp2tok.data -= tok_start
        tok2wp.data -= wp_start
        return wp2tok, tok2wp


def _align_batch(A: List[List[str]], B: List[List[str]]) -> Tuple[Ragged, Ragged]:
    if len(A) != len(B):
        raise ValueError("Cannot align batches of different sizes.")
    a_stride = max((len(a) for a in A), default=0)
    b_stride = max((len(b) for b in B), default=0)
    A2B = []
    B2A = []
    a2b_lengths = []
    b2a_lengths = []
    a_start = 0
    b_start = 0
    for i, (a, b) in enumerate(zip(A, B)):
        a2b, b2a = tokenizations.get_alignments(a, b)
        for b_js in a2b:
            A2B.extend([b_start + b_j for b_j in b_js])
            a2b_lengths.append(len(b_js))
        for a_js in b2a:
            B2A.extend([a_start + a_j for a_j in a_js])
            b2a_lengths.append(len(a_js))
        b_start += b_stride
        a_start += a_stride
    return (
        Ragged(numpy.array(A2B, dtype="i"), numpy.array(a2b_lengths, dtype="i")),
        Ragged(numpy.array(B2A, dtype="i"), numpy.array(b2a_lengths, dtype="i")),
    )
