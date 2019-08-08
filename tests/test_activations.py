import pytest
import numpy
from spacy_pytorch_transformers.activations import Activations


@pytest.fixture
def model(nlp):
    return nlp.get_pipe("pytt_tok2vec").model._model

def test_act_blank():
    acts = Activations.blank()
    assert acts.lh.data.size == 0
    assert acts.lh.lengths == []
    assert acts.po.data.size == 0
    assert acts.po.lengths == []
    assert not acts.has_lh
    assert not acts.has_po
