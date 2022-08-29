from typing import Dict, List, Tuple, Callable, Optional
from spacy.tokens import Span, Token
from thinc.api import Ops
from thinc.types import Ragged, Floats2d, Ints2d

def apply_alignment(
    ops: Ops, align: Ragged, X: Floats2d
) -> Tuple[Ragged, Callable]: ...
def get_token_positions(spans: List[Span]) -> Dict[Token, int]: ...
def get_alignment_via_offset_mapping(
    spans: List[Span],
    offset_mapping: Ints2d,
) -> Ragged: ...
def get_alignment(
    spans: List[Span],
    wordpieces: List[List[str]],
    special_tokens: Optional[List[str]] = None,
) -> Ragged: ...
def get_span2wp_from_offset_mapping(
    span: Span,
    wp_char_offsets: Tuple[int],
) -> List[List[int]]: ...
