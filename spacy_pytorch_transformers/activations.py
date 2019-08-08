from thinc.neural.util import get_array_module
from dataclasses import dataclass
import numpy

from .util import List, Array, Union, Any
from .util import pad_batch, ensure3d


@dataclass
class Activations:
    lh: Array
    po: Array
    ah: List[Array]
    aa: List[Array]
    wlengths: List[int]
    slengths: List[int]

    @classmethod
    def blank(cls, *, xp=numpy):
        return cls(
            xp.zeros((0, 0), dtype="f"), xp.zeros((0, 0), dtype="f"),
            [], [], [], []
        )

    @classmethod
    def join(cls, sub_acts: List["Activations"]) -> "Activations":
        """Concatenate activations from subsequences."""
        xp = get_array_module(sub_acts[0].lh)
        lh: Array = xp.vstack([x.lh for x in sub_acts])
        po: Array = xp.vstack([x.po for x in sub_acts])
        # Transpose the lists, so that the inner list items refer
        # to the subsequences. Then we can vstack those.
        ah = list(map(xp.vstack, zip(*[x.ah for x in sub_acts])))
        # TODO: Support aa
        aa = []
        return cls(lh, po, ah, aa)

    @classmethod
    def pad_batch(cls, batch: List["Activations"], *, to: int = 0) -> "Activations":
        if not batch:
            return Activations.blank()
        xp = get_array_module(batch[0])
        lh = pad_batch([x.lh for x in batch], xp=xp, to=to, axis=-2)
        po = pad_batch([x.po for x in batch], xp=xp)
        # Transpose the lists, and then pad_batch the items
        ah = [
            pad_batch(list(seq), xp=xp, to=to, axis=1)
            for seq in zip(*[x.ah for x in batch])
        ]
        # TODO: Support aa
        aa = []
        return Activations(lh, po, ah, aa)

    def __len__(self) -> int:
        return len(self.lh)

    def get_slice(self, batch, word_slice, sent_slice) -> "Activations":
        lh = ensure3d(self.lh[batch, word_slice])
        po = ensure3d(self.po[batch, sent_slice] if self.has_po else self.po)
        ah = [ensure3d(self.ah[i][batch, word_slice]) for i in range(len(self.ah))]
        # TODO: Support aa
        aa = []
        return Activations(lh, po, ah, aa)

    def split(self, ops: Any, lengths: List[int]) -> List["Activations"]:
        """Split into a list of Activation objects."""
        lh = ops.unflatten(self.lh, lengths)
        po = ops.unflatten(self.po, lengths)
        print("lh", [x.shape for x in lh], "po", [x.shape for x in po])
        # Transpose the lists, so that the outer list refers to the subsequences
        if self.ah:
            ah = list(zip(*[ops.unflatten(x, lengths) for x in self.ah]))
        else:
            ah = [[] for _ in lengths]
        # TODO: Support aa
        aa = [[] for _ in lengths]
        assert len(lh) == len(po) == len(ah) == len(aa)
        # Make an Activations object for each subsequence.
        all_args = zip(lh, po, ah, aa)
        return [Activations(*args) for args in all_args]

    def untruncate(self, to: int) -> "Activations":
        raise NotImplementedError

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
