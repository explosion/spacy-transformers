import pytest
import numpy


@pytest.fixture
def model(nlp):
    return nlp.get_pipe("pytt_tok2vec").model._model


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
