from collections import namedtuple
import torch
from thinc.extra.wrappers import PyTorchWrapper, torch2xp, xp2torch

from .util import get_pytt_model, Activations


class PyTT_Wrapper(PyTorchWrapper):
    """Wrap a PyTorch-Transformers model for use in Thinc."""

    @classmethod
    def from_pretrained(cls, name):
        self = cls(name)
        self._model = self._model.from_pretrained(name)
        return self

    def __init__(self, name):
        model = get_pytt_model(name)
        PyTorchWrapper.__init__(self, model)

    @property
    def nO(self):
        return self._model.config.hidden_size

    def begin_update(self, ids, drop=None):
        ids = xp2torch(ids)
        is_training = self._model.training
        if drop is None:
            self._model.eval()
            y_var = self._model(ids)
        else:
            self._model.train()
            y_var = self._model(ids)
        self._model.training = is_training
        output = Activations.from_pytt(y_var, is_grad=False)
        assert output.has_last_hidden_state

        def backward_pytorch(dy_data, sgd=None):
            y_for_bwd = []
            dy_for_bwd = []
            if dy_data.has_last_hidden_state:
                dy_for_bwd.append(xp2torch(dy_data.last_hidden_state))
                y_for_bwd.append(y_var[0])
            if dy_data.has_pooler_output:
                dy_for_bwd.append(xp2torch(dy_data.pooler_output))
                y_for_bwd.append(y_var[1])
            if dy_data.has_all_hidden_states:
                raise ValueError(
                    "Gradients on all hidden states not supported yet.")
            if dy_data.has_all_attentions:
                raise ValueError(
                    "Gradients on all attentions not supported yet.")
            torch.autograd.backward(y_for_bwd, grad_tensors=dy_for_bwd)
            if sgd is not None:
                if self._optimizer is None:
                    self._optimizer = self._create_optimizer(sgd)
                self._optimizer.step()
                self._optimizer.zero_grad()
            return None

        self._model.eval()
        return output, backward_pytorch
