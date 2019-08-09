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
        shape = (len(self.lengths), pad_to) + self.data.shape[1:]
        values = self.xp.zeros(shape, dtype=self.dtype)
        if self.data.size == 0:
            return values
        # Slightly convoluted implementation here, to do the operation in one
        # and avoid the loop
        mask = lengths2mask(self.lengths)
        values = values.reshape((len(self.lengths) * pad_to,) + self.data.shape[1:])
        values[mask >= 1] = self.data
        values = values.reshape(shape)
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
    def xp(self):
        return self.lh.xp

    @property
    def has_lh(self) -> bool:
        return bool(self.lh.data.size)

    @property
    def has_po(self) -> bool:
        return bool(self.po.lengths)
