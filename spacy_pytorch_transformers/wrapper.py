from thinc.extra.wrappers import PyTorchWrapper, xp2torch
from pytorch_transformers.optimization import AdamW
import torch.autograd
import torch.nn.utils.clip_grad
import torch
from typing import Tuple, Callable, Any
from thinc.neural.optimizers import Optimizer

from .util import get_pytt_model, Activations
from .util import Array, Dropout

FINE_TUNE = True
GRAD_CLIP_FACTOR = 100
LEARN_RATE_FACTOR = 100
CONFIG = {"output_hidden_states": True, "output_attentions": True}


class PyTT_Wrapper(PyTorchWrapper):
    """Wrap a PyTorch-Transformers model for use in Thinc."""

    _model: Any
    _optimizer: Any

    @classmethod
    def from_pretrained(cls, name):
        self = cls(name)
        self._model = self._model.from_pretrained(name, **CONFIG)
        return self

    def __init__(self, name):
        model = get_pytt_model(name)
        PyTorchWrapper.__init__(self, model)

    @property
    def nO(self):
        return self._model.config.hidden_size

    def begin_update(
        self, ids: Array, drop: Dropout = None
    ) -> Tuple[Activations, Callable[..., None]]:
        ids = xp2torch(self.ops.asarray(ids))
        is_training = self._model.training
        if drop is None:
            self._model.eval()
            y_var = self._model(ids)
        else:
            self._model.train()
            y_var = self._model(ids)
        self._model.training = is_training
        output = Activations.from_pytt(y_var, is_grad=False)
        assert output.lh is not None

        def backward_pytorch(d_output: Activations, sgd: Optimizer = None) -> None:
            y_for_bwd = []
            dy_for_bwd = []
            if d_output.has_lh:
                dy_for_bwd.append(xp2torch(d_output.lh))
                y_for_bwd.append(y_var[0])
            if d_output.has_po:
                raise ValueError("Gradients on all hidden states not supported yet.")
            if d_output.has_ah:
                raise ValueError("Gradients on all hidden states not supported yet.")
            if d_output.has_aa:
                raise ValueError("Gradients on all attentions not supported yet.")
            if FINE_TUNE:
                torch.autograd.backward(y_for_bwd, grad_tensors=dy_for_bwd)
                if sgd is not None:
                    if self._optimizer is None:
                        self._optimizer = self._create_optimizer(sgd)
                    if sgd.max_grad_norm:
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            self._model.parameters(),
                            sgd.max_grad_norm / GRAD_CLIP_FACTOR,
                        )
                    optimizer = self._optimizer
                    optimizer.step()
                    optimizer.zero_grad()
            return None

        self._model.eval()
        return output, backward_pytorch

    def _create_optimizer(self, sgd):
        optimizer = AdamW(
            self._model.parameters(),
            lr=sgd.alpha / LEARN_RATE_FACTOR,
            betas=(sgd.b1, sgd.b2),
        )
        return optimizer
