from typing import List, Tuple, Optional, Dict

_OffsetsT = List[Optional[Tuple[int, int]]]
_AlignmentT = List[List[int]]


def align_offsets(offsets1: _OffsetsT, offsets2: _OffsetsT) -> _AlignmentT:
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
