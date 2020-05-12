import numpy
from dataclasses import dataclass
from typing import List, Tuple
import tokenizations
from thinc.api import Ragged


@dataclass
class BatchAlignment:
    """Alignment for a batch of texts between wordpieces and tokens."""

    tok2trf: Ragged
    trf2tok: Ragged
    trf_lengths: List[int]
    tok_lengths: List[int]

    def from_strings(cls, tok: List[List[str]], trf: List[List[str]]):
        # TODO: This needs to take into account that the same token can
        # be in multiple spans.
        tok2trf, trf2tok = _align_batch(tok, trf)
        trf_lengths = [len(x) for x in trf]
        tok_lengths = [len(x) for x in tok]
        return cls(
            tok2trf=tok2trf,
            trf2tok=trf2tok,
            trf_lengths=trf_lengths,
            tok_lengths=tok_lengths
        )

    def slice(self, start: int, end: int) -> Tuple[Ragged, Ragged]:
        """Extract the alignment for a subset of the batch, adjusting the
        indices accordingly."""
        tok_start = sum(self.tok_lengths[:start])
        tok_end = sum(self.tok_lengths[:end])
        trf_start = sum(self.trf_lengths[:start])
        trf_end = sum(self.trf_lengths[:end])
        # Get the slice, and adjust the data so they point to the right part
        # of the target array. We're trusting that nothing will point past the
        # ends.
        tok2trf = self.tok2trf[tok_start:tok_end]
        trf2tok = self.trf2tok[trf_start:trf_end]
        tok2trf.data -= trf_start
        trf2tok.data -= tok_start
        return tok2trf, trf2tok


def _align_batch(spans: List[Span], wordpieces: List[List[str]]) -> Tuple[Ragged, Ragged]:
    if len(A) != len(B):
        raise ValueError("Cannot align batches of different sizes.")
    token_positions: Dict[Tuple[int, int], int] = {}
    flat_tokens = []
    for span in spans:
        for token in span:
            key = (id(span.doc), token.start_char)
            if key not in token_positions:
                token_positions[key] = len(token_positions) 
    A2B = [[] for _ in range(len(token_positions))]
    B2A = [[] for _ in range(sum(len(wp) for wp in wordpieces))]
    a2b_lengths = []
    b2a_lengths = []
    a_start = 0
    b_start = 0
    for i, (span, wp_texts) in enumerate(zip(spans, wordpieces)):
        tok_texts = [token.text for token in span]
        span2wp, wp2span = tokenizations.get_alignments(tok_text, wp_texts)
        for token, wp_js in zip(span2wp, span):
            key = (id(span.doc), token.start_char)
            position = token_positions[key]
            A2B[position].extend(wp_js)



    for i, (a, b) in enumerate(zip(A, B)):
        a2b, b2a = tokenizations.get_alignments(a, b)
        for b_js in a2b:
            A2B.extend([b_start + b_j for b_j in b_js])
            a2b_lengths.append(len(b_js))
        for a_js in b2a:
            B2A.extend([a_start + a_j for a_j in a_js])
            b2a_lengths.append(len(a_js))
        a_start += len(a)
        b_start += len(b)
    assert len(a2b_lengths) == sum(len(a) for a in A)
    assert len(b2a_lengths) == sum(len(b) for b in B)
    return (
        Ragged(numpy.array(A2B, dtype="i"), numpy.array(a2b_lengths, dtype="i")),
        Ragged(numpy.array(B2A, dtype="i"), numpy.array(b2a_lengths, dtype="i")),
    )
