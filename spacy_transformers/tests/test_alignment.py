from typing import List, Tuple, Optional
from dataclasses import dataclass
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pytest
from .._align import align_offsets, align_spans, align_docs


@dataclass
class AlignmentCase:
    string: str
    offsets1: List[Optional[Tuple[int, int]]]
    offsets2: List[Optional[Tuple[int, int]]]
    a2b: List[List[int]]
    b2a: List[List[int]]

    @property
    def words1(self):
        words = []
        for entry in self.offsets1:
            if entry is not None:
                start, end = entry
                words.append(self.string[start:end])
        return words

    def make_doc(self):
        return Doc(Vocab(), words=self.words1, spaces=[False] * len(self.words1))

    def make_spans(self):
        doc = self.make_doc()
        return [doc[:]]


CASES = {
    "match": AlignmentCase(
        "ab", [(0, 1), (1, 2)], [(0, 1), (1, 2)], [[0], [1]], [[0], [1]]
    ),
    "multi1": AlignmentCase("ab", [(0, 2)], [(0, 1), (1, 2)], [[0, 1]], [[0], [0]]),
    "multi2": AlignmentCase("ab", [(0, 1), (1, 2)], [(0, 2)], [[0], [0]], [[0, 1]]),
    "mix1": AlignmentCase(
        "abc",
        [(0, 2), (2, 3)],
        [(0, 1), (1, 2), (2, 3)],
        [[0, 1], [2]],
        [[0], [0], [1]],
    ),
    "nones": AlignmentCase(
        "abc",
        [None, (0, 2), (2, 3), None],
        [(0, 1), (1, 2), (2, 3)],
        [[], [0, 1], [2], []],
        [[1], [1], [2]],
    ),
}


@pytest.mark.parametrize("case_id", list(CASES.keys()))
def test_align_offsets(case_id):
    case = CASES[case_id]
    predicted_a2b = align_offsets(case.offsets1, case.offsets2)
    assert predicted_a2b == case.a2b
    predicted_b2a = align_offsets(case.offsets2, case.offsets1)
    assert predicted_b2a == case.b2a


@pytest.mark.parametrize("case_id", list(CASES.keys()))
def test_align_spans(case_id):
    # This is a pretty bad test, it doesn't check any of the likely logic errors.
    # But at least it does run the code?
    case = CASES[case_id]
    spans = case.make_spans()
    alignments = align_spans(spans, [case.offsets2])
    assert len(alignments) == len(spans)
    for i, span in enumerate(spans):
        assert len(alignments[i]) == len(span)
        for align in alignments[i]:
            for entry in align:
                assert entry[0] == i


@pytest.mark.parametrize("case_id", list(CASES.keys()))
def test_align_docs(case_id):
    # This is a pretty bad test, it doesn't check any of the likely logic errors.
    # But at least it does run the code?
    case = CASES[case_id]
    spans = case.make_spans()
    span_alignments = align_spans(spans, [case.offsets2])
    doc_alignments = align_docs(spans, span_alignments)
    assert len(doc_alignments) == 1
    assert doc_alignments == span_alignments
