from collections import namedtuple
import torch
from thinc.extra.wrappers import PyTorchWrapper, torch2xp, xp2torch

from .util import get_pytt_model


class PyTT_Wrapper(PyTorchWrapper):
    """Wrap a PyTorch-Transformers model for use in Thinc."""

    @classmethod
    def from_pretrained(cls, name):
        self = cls(name)
        self._model = self._model.from_pretrained(name)
        return self

    def __init__(self, name, out_cols=("last_hidden_state", "pooler_output")):
        model = get_pytt_model(name)
        PyTorchWrapper.__init__(self, model)
        self.out_cols = out_cols

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
        converted_outputs = []
        for var in y_var[: len(self.out_cols)]:
            if isinstance(var, tuple):
                converted_outputs.append(var)
            else:
                converted_outputs.append(torch2xp(var))
        output = namedtuple("pytt_outputs", self.out_cols)(*converted_outputs)

        def backward_pytorch(dy_data, sgd=None):
            y_for_bwd = []
            dy_for_bwd = []
            for y, dy in zip(y_var, dy_data):
                if dy is not None:
                    dy_for_bwd.append(xp2torch(dy))
                    y_for_bwd.append(y)
            torch.autograd.backward(y_for_bwd, grad_tensors=dy_for_bwd)
            if sgd is not None:
                if self._optimizer is None:
                    self._optimizer = self._create_optimizer(sgd)
                self._optimizer.step()
                self._optimizer.zero_grad()
            return None

        assert output.last_hidden_state is not None
        self._model.eval()
        return output, backward_pytorch
