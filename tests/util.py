from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil


@contextmanager
def make_tempdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))
