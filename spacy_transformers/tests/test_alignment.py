from typing import List, Tuple, Optional
from dataclasses import dataclass
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pytest
from .._align import BatchAlignment


class AlignmentCase:
    A: List[List[str]]
    B: List[List[str]]
    a2b: List[int]
    b2a: List[int]



CASES = {
    "match": AlignmentCase([["a", "b"]], [["a", "b"]], [0, 1], [0, 1]),
    "multi1": AlignmentCase([["a", "b"]], [["ab"]], , [0, 0], [0]),
    "multi2": AlignmentCase([["ab"]], [["a", "b"]], [0], [0, 0]),
    "mix1": AlignmentCase([["ab", "c"]], [["a", "bc"]], [0, 1], [0, 1])
}




words1 = ["a", "b"]
words2 = ["ab"]

feats1 = [
    [1, -1, 1],
    [0, 0, 0]
]

feats2 = [
    [0.5, -0.5, 0.5]
]

alignment = [0, 1]

for i, j in enumerate(alignment):
    feats2[j] += feats1[i]


words1 = ["ab"]
words2 = ["a", "b"]

feats1 = [
    [1, -1, 1],
]

feats2 = [
    [1, -1, 1],
    [1, -1, 1],
]

alignment = [0, 1]


