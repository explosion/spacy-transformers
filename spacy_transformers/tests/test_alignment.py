from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import pytest
from .._align import align_offsets


@dataclass
class AlignmentCase:
    string: str
    offsets1: List[Optional[Tuple[int, int]]]
    offsets2: List[Optional[Tuple[int, int]]]
    a2b: List[List[int]] 
    b2a: List[List[int]]


CASES = {
    "match": AlignmentCase(
        "ab",
        [(0, 1), (1, 2)],
        [(0, 1), (1, 2)],
        [[0], [1]],
        [[0], [1]]
    ),
    "multi1": AlignmentCase(
        "ab",
        [(0, 2)],
        [(0, 1), (1, 2)],
        [[0, 1]],
        [[0], [0]]
    ),
    "multi2": AlignmentCase(
        "ab",
        [(0, 1), (1, 2)],
        [(0, 2)],
        [[0], [0]],
        [[0, 1]]
    ),
    "mix1": AlignmentCase(
        "abc",
        [(0, 2), (2, 3)],
        [(0, 1), (1, 2), (2, 3)],
        [[0, 1], [2]],
        [[0], [0], [1]]
    ),
    "nones": AlignmentCase(
        "abc",
        [None, (0, 2), (2, 3), None],
        [(0, 1), (1, 2), (2, 3)],
        [[], [0, 1], [2], []],
        [[1], [1], [2]]
    ),
}

@pytest.mark.parametrize("case_id", list(CASES.keys()))
def test_align_offsets(case_id):
    case = CASES[case_id]
    predicted_a2b = align_offsets(case.offsets1, case.offsets2)
    assert predicted_a2b == case.a2b
    predicted_b2a = align_offsets(case.offsets2, case.offsets1)
    assert predicted_b2a == case.b2a
