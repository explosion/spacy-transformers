from thinc.neural.util import get_array_module
from dataclasses import dataclass
import numpy

from .util import List, Array
from .util import lengths2mask


@dataclass
class RaggedArray:
    data: Array
    lengths: List[int]

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def xp(self):
        return get_array_module(self.data)

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def blank(cls, xp=numpy) -> "RaggedArray":
        return RaggedArray(xp.zeros((0,), dtype="f"), [])

    @classmethod
    def from_truncated(cls, square: Array, lengths: List[int]) -> "RaggedArray":
        if len(lengths) != square.shape[0]:
            raise ValueError("Truncated array must have shape[0] == len(lengths)")
        width = square.shape[1]
        max_len = max(lengths, default=0)
        extra_dims = square.shape[2:]
        if width == max_len:
            return RaggedArray(square, lengths)
        elif width > max_len:
            raise ValueError("Expected width < max_len. Got {width} > {max_len}")
        xp = get_array_module(square)
        expanded = xp.zeros((sum(lengths),) + extra_dims, dtype=square.dtype)
        # TODO: I know there's a way to do this without the loop :(. Escapes
        # me currently.
        start = 0
        for i, length in enumerate(lengths):
            # We could have a row that's actually shorter than the width,
            # if the array was padded. Make sure we don't get junk values.
            row_width = min(width, length)
            expanded[start : start + row_width] = square[i, :row_width]
            start += length
        return cls(expanded, lengths)

    @classmethod
    def from_padded(cls, padded: Array, lengths: List[int]) -> "RaggedArray":
        if max(lengths, default=0) > padded.shape[1]:
            return cls.from_truncated(padded, lengths)
        mask = lengths2mask(lengths)
        assert sum(mask) == sum(lengths)
        all_rows = padded.reshape((-1,) + padded.shape[2:])
        xp = get_array_module(all_rows)
        data = xp.ascontiguousarray(all_rows[mask])
        assert data.shape[0] == sum(lengths)
        return cls(data, lengths)

    def to_padded(self, *, value=0, to: int = -1) -> Array:
        assert sum(self.lengths) == self.data.shape[0]
        max_len = max(self.lengths, default=0)
        if to >= 1 and to < max_len:
            raise ValueError(f"Cannot pad to {to}: Less than max length {max_len}")
        to = max(to, max_len)
        # Slightly convoluted implementation here, to do the operation in one
        # and avoid the loop
        shape = (len(self.lengths), to) + self.data.shape[1:]
        values = self.xp.zeros(shape, dtype=self.dtype)
        if value != 0:
            values.fill(value)
        if self.data.size == 0:
            return values
        mask = lengths2mask(self.lengths)
        values = values.reshape((len(self.lengths) * to,) + self.data.shape[1:])
        values[mask] = self.data
        values = values.reshape(shape)
        return values

    def get(self, i: int) -> Array:
        start = sum(self.lengths[:i])
        end = start + self.lengths[i]
        return self.data[start:end]


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
        return bool(self.po.data.size)
