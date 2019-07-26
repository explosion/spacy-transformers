from collections import namedtuple
import pytorch_transformers as pytt
import torch

from thinc.extra.wrappers import PyTorchWrapper, torch2xp, xp2torch


def get_pytt_config(name):
    return pytt.BertConfig.from_pretrained(name)


def get_pytt_model(name):
    return pytt.BertModel(get_pytt_config(name))


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
        self._model.train()
        y_var = self._model(ids)
        output = namedtuple("pytt_outputs", self.out_cols)(*map(torch2xp, y_var))

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

        return output, backward_pytorch
