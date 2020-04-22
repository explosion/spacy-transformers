from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch


@dataclass
class TokensPlus:
    """Dataclass to hold the output of the Huggingface 'batch_encode_plus' method."""
    input_words: List[List[str]]
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_len: List[int]
    offsets: List[List[Optional[Tuple[int, int]]]]
    overflowing_tokens: Optional[torch.Tensor] = None
    num_truncated_tokens: Optional[torch.Tensor] = None
    special_tokens_mask: Optional[torch.Tensor] = None


@dataclass
class TransformerOutput:
    tokens: TokensPlus
    tensors: Tuple[FloatsXd]
    spans: List[Span]

    @property
    def width(self) -> int:
        return self.tensors[-1].shape[-1]
