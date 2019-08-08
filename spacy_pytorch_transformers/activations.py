from thinc.neural.util import get_array_module
from dataclasses import dataclass
import numpy

from .util import List, Array, Union
from .util import lengths2mask, pad_batch


@dataclass
class RaggedArray:
    data: Array
    lengths: List[int]

    @classmethod
    def blank(cls, xp=numpy) -> "RaggedArray":
        return RaggedArray(xp.zeros((0,), dtype="f"), [])

    @classmethod
    def from_padded(cls, padded: Array, lengths: List[int]) -> "RaggedArray":
        mask = lengths2mask(lengths)
        all_rows = padded.reshape((-1, padded.shape[-1]))
        xp = get_array_module(all_rows)
        data = xp.ascontiguousarray(all_rows[mask])
        return cls(data, lengths)

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def xp(self):
        return get_array_module(self.data)

    @property
    def dtype(self):
        return self.data.dtype
    
    def to_padded(self, value=0) -> Array:
        return pad_batch([self.data], to=max(self.lengths, default=0), value=value)


@dataclass
class Activations:
    lh: RaggedArray
    po: RaggedArray

    @classmethod
    def blank(cls, *, xp=numpy):
        return cls(RaggedArray.blank(xp=xp), RaggedArray.blank(xp=xp))

    @property
    def xp(self) -> Union["numpy", "cupy"]:
        return self.lh.xp

    @property
    def has_lh(self) -> bool:
        return bool(self.lh.lengths)

    @property
    def has_po(self) -> bool:
        return bool(self.po.lengths)
