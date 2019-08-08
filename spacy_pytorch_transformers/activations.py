from thinc.neural.util import get_array_module
from dataclasses import dataclass
import numpy

from .util import List, Array, Union, Any
from .util import pad_batch, ensure3d, lengths2mask


@dataclass
class RaggedArray:
    data: Array
    lengths: List[int]

    @classmethod
    def from_padded(cls, padded, lengths):
        mask = lengths2mask(lengths)
        all_rows = padded.reshape((-1, padded.shape[-1]))
        xp = get_array_module(all_rows)
        data = xp.ascontiguousarray(all_rows[mask])
        return cls(data, lengths)

    @property
    def xp(self):
        return get_array_module(self.data)

    @property
    def dtype(self):
        return self.data.dtype


@dataclass
class Activations:
    lh: RaggedArray
    po: RaggedArray

    @classmethod
    def blank(cls, *, xp=numpy):
        return cls(RaggedArray.blank(xp=xp), RaggedArray.blank(xp=xp))

    @property
    def xp(self) -> Union["numpy", "cupy"]:
        return get_array_module(self.lh)

    @property
    def has_lh(self) -> bool:
        return bool(self.lh.size)

    @property
    def has_po(self) -> bool:
        return bool(self.po.size)

    @property
    def has_ah(self) -> bool:
        return bool(sum(len(x) for x in self.ah))

    @property
    def has_aa(self) -> bool:
        return bool(sum(len(x) for x in self.aa))
