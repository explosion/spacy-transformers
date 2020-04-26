from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch
from thinc.types import FloatsXd
from thinc.api import Ops, torch2xp, get_current_ops
from spacy.tokens import Span


@dataclass
class TokensPlus:
    """Dataclass to hold the output of the Huggingface 'batch_encode_plus' method."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: List[List[Optional[Tuple[int, int]]]]
    token_type_ids: Optional[torch.Tensor] = None
    overflowing_tokens: Optional[torch.Tensor] = None
    num_truncated_tokens: Optional[torch.Tensor] = None
    special_tokens_mask: Optional[torch.Tensor] = None

    @classmethod
    def empty(cls) -> "TokensPlus":
        return cls(
            input_ids=torch.zeros(0, 0),
            attention_mask=torch.zeros(0, 0),
            offset_mapping=[],
        )


@dataclass
class TransformerOutput:
    tokens: TokensPlus
    tensors: Tuple[torch.Tensor]
    spans: List[Span]
    ops: Ops

    @classmethod
    def empty(cls) -> "TransformerOutput":
        ops = get_current_ops()
        return cls(tokens=TokensPlus.empty(), tensors=[], spans=[], ops=ops)

    @property
    def docs(self):
        seen = set()
        docs = []
        for span in self.spans:
            key = id(span.doc)
            if key not in seen:
                docs.append(span.doc)
                seen.add(key)
        return docs

    @property
    def width(self) -> int:
        return self.tensors[-1].shape[-1]

    @property
    def arrays(self) -> List[FloatsXd]:
        return [torch2xp(tensor) for tensor in self.tensors]
