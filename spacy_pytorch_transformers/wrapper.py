from thinc.extra.wrappers import PyTorchWrapper, xp2torch
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import pytorch_transformers as pytt
import torch.autograd
import torch.nn.utils.clip_grad
import torch
from typing import Tuple, Callable, Any
from thinc.neural.optimizers import Optimizer
from thinc.neural.util import get_array_module

from .util import get_pytt_model, Activations
from .util import Array, Dropout

FINE_TUNE = True
CONFIG = {"output_hidden_states": True, "output_attentions": True}


class PyTT_Wrapper(PyTorchWrapper):
    """Wrap a PyTorch-Transformers model for use in Thinc."""

    _model: Any
    _optimizer: Any
    _lr_schedule: Any

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

    @property
    def max_length(self):
        return self._model.config.max_position_embeddings

    def predict(self, ids: Array):
        model_kwargs = self.get_model_kwargs(ids)
        is_training = self._model.training
        self._model.training = False
        with torch.no_grad():
            y_var = self._model(**model_kwargs)
        self._model.training = is_training
        return Activations.from_pytt(y_var, is_grad=False)

    def begin_update(
        self, ids: Array, drop: Dropout = 0.0
    ) -> Tuple[Activations, Callable[..., None]]:
        if drop is None:
            # "drop is None" indicates prediction. It's one of the parts of
            # Thinc's API I'm least happy with...
            return self.predict(ids), lambda dY, sgd=None: None
        # Prepare all the model arguments, including the attention mask
        model_kwargs = self.get_model_kwargs(ids)
        is_training = self._model.training
        self._model.train()
        y_var = self._model(**model_kwargs)
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
                dy_for_bwd.append(xp2torch(d_output.po).reshape((d_output.po.shape[0], d_output.po.shape[2])))
                y_for_bwd.append(y_var[1])
            if d_output.has_ah:
                raise ValueError("Gradients on all hidden states not supported yet.")
            if d_output.has_aa:
                raise ValueError("Gradients on all attentions not supported yet.")
            if FINE_TUNE:
                torch.autograd.backward(y_for_bwd, grad_tensors=dy_for_bwd)
                if sgd is not None:
                    if self._optimizer is None:
                        self._optimizer = self._create_optimizer(sgd)
                        self._optimizer.zero_grad()
                    if sgd.max_grad_norm:
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            self._model.parameters(),
                            sgd.max_grad_norm 
                        )
                    optimizer = self._optimizer
                    optimizer.lr = sgd.alpha
                    optimizer.step()
                    optimizer.zero_grad()
            return None

        return output, backward_pytorch

    def get_model_kwargs(self, ids):
        if isinstance(ids, list):
            ids = numpy.array(ids, dtype=numpy.int_)
        # Calculate "attention mask" for BERT and  XLNet, but not GPT2 (sigh)
        xp = get_array_module(ids)
        neg_idx = ids < 0
        ids[neg_idx] = 0
        if isinstance(self._model, (pytt.BertModel, pytt.XLNetModel)):
            mask = xp.ones(ids.shape, dtype="f")
            mask[neg_idx] = 0
            segment_ids = xp.zeros(ids.shape, dtype=ids.dtype)
            output = {"input_ids": ids, "attention_mask": mask, "token_type_ids": segment_ids}
        else:
            output = {"input_ids": ids}
        return {key: xp2torch(val) for key, val in output.items()}

    def _create_optimizer(self, sgd):
        optimizer = AdamW(
            self._model.parameters(),
            lr=sgd.alpha,
            eps=sgd.eps,
            betas=(sgd.b1, sgd.b2),
        )
        optimizer.zero_grad()
        return optimizer
