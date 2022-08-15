from typing import List, Tuple, Callable, Optional
from spacy.tokens import Span, Token
from thinc.api import Ops
from thinc.types import Ragged, Floats2d


def apply_alignment(ops: Ops, align: Ragged, X: Floats2d) -> Tuple[Ragged, Callable]: ...

def get_token_positions(spans: List[Span]) -> Dict[Token, int]: ...

def get_alignment_via_offset_mapping(
    spans: List[Span],
    offset_mapping: List[Tuple[int]],
) -> Ragged: ...

def get_alignment(
    spans: List[Span],
    wordpieces: List[List[str]],
    special_tokens: Optional[List[str]] = None,
) -> Ragged: ...
