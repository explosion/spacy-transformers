from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil
import numpy


@contextmanager
def make_tempdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))


def is_valid_tensor(tensor):
    return tensor is not None and numpy.nonzero(tensor) and tensor.size != 0
