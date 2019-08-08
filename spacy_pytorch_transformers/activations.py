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
        pad_to = max(self.lengths, default=0)
        values = self.xp.zeros((len(self.lengths), pad_to), dtype=self.dtype)
        values[:] = value
        mask = lengths2mask(self.lengths)
        start = 0
        for i, length in enumerate(self.lengths):
            values[i, :length] = self.data[start:start+length]
            start += length
        return values

    def get(self, i: int) -> Array:
        start = sum(self.lengths[:i])
        end = start + self.lengths[i]
        return self.data[start : end]


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
