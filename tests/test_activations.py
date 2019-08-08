import pytest
import numpy
from spacy_pytorch_transformers.activations import Activations


@pytest.fixture
def model(nlp):
    return nlp.get_pipe("pytt_tok2vec").model._model

def test_act_blank():
    acts = Activations.blank()
    assert acts.lh.ndim == 3
    assert acts.lh.size == 0
    assert acts.po.ndim == 3
    assert acts.po.size == 0
    assert acts.ah == []
    assert acts.aa == []
    assert not acts.has_lh
    assert not acts.has_po
    assert not acts.has_ah
    assert not acts.has_aa


@pytest.mark.parametrize("ids", [[[10, 5], [7, 9]]])
def test_act_fields(name, model, ids):
    ids = numpy.array(ids, dtype=numpy.int_)
    acts = model(ids)
    assert len(acts) == len(ids)
    if acts.has_po:
        assert len(acts.po) >= 1
    if acts.has_ah:
        assert len(acts.ah) >= 1
    if acts.has_aa:
        assert len(acts.aa) >= 1

# TODO: Test acts.split()
# TODO: Test acts.get_slice()
# TODO: Test Acts.pad_batch()
# TODO: Test Acts.join()
