from thinc.extra.wrappers import PyTorchWrapper, xp2torch
from pytorch_transformers.optimization import AdamW
import pytorch_transformers as pytt
import torch.autograd
import torch.nn.utils.clip_grad
import torch
from typing import Tuple, Callable, Any
from thinc.neural.optimizers import Optimizer
import numpy
from torchcontrib.optim import SWA

from .util import get_pytt_model, get_pytt_config, Activations
from .util import Array, Dropout

FINE_TUNE = True
CONFIG = {"output_hidden_states": True, "output_attentions": True}


class PyTT_Wrapper(PyTorchWrapper):
    """Wrap a PyTorch-Transformers model for use in Thinc."""

    _model: Any
    _optimizer: Any
    _lr_schedule: Any
    cfg: dict

    @classmethod
    def from_pretrained(cls, name):
        config_cls = get_pytt_config(name)
        model_cls = get_pytt_model(name)
        config = config_cls.from_pretrained(name)
        model = model_cls.from_pretrained(name, **CONFIG)
        self = cls(name, config.to_dict(), model)
        self.cfg.update(self.pytt_model.config.to_dict())
        return self

    def __init__(self, name, config, model):
        PyTorchWrapper.__init__(self, model)
        self.cfg = dict(config)

    @property
    def nO(self):
        if "hidden_size" in self.cfg:
            # BERT
            return self.cfg["hidden_size"]
        elif "n_embd" in self.cfg:
            # GPT2
            return self.cfg["n_embd"]
        elif "d_model" in self.cfg:
            # XLNet
            return self.cfg["d_model"]
        elif hasattr(self.pytt_model, "dim"):
            # XLM
            return self.pytt_model.dim
        else:
            keys = ", ".join(self.cfg.keys())
            raise ValueError(f"Unexpected config. Keys: {keys}")

    @property
    def pytt_model(self):
        return self._model

    @property
    def max_length(self):
        return self.cfg["max_position_embeddings"]

    def predict(self, ids: Array):
        self._model.eval()
        model_kwargs = self.get_model_kwargs(ids)
        with torch.no_grad():
            if hasattr(self._optimizer, "swap_swa_sgd"):
                self._optimizer.swap_swa_sgd()
            y_var = self._model(**model_kwargs)
            if hasattr(self._optimizer, "swap_swa_sgd"):
                self._optimizer.swap_swa_sgd()
        return Activations.from_pytt(y_var, is_grad=False)

    def begin_update(
        self, ids: Array, drop: Dropout = 0.0
    ) -> Tuple[Activations, Callable[..., None]]:
        if drop is None:
            # "drop is None" indicates prediction. It's one of the parts of
            # Thinc's API I'm least happy with...
            return self.predict(ids), lambda dY, sgd=None: None
        self._model.train()
        # Prepare all the model arguments, including the attention mask
        model_kwargs = self.get_model_kwargs(ids)
        y_var = self._model(**model_kwargs)
        output = Activations.from_pytt(y_var, is_grad=False)
        assert output.lh is not None

        def backward_pytorch(d_output: Activations, sgd: Optimizer = None) -> None:
            y_for_bwd = []
            dy_for_bwd = []
            if d_output.has_lh:
                dy_for_bwd.append(xp2torch(d_output.lh))
                y_for_bwd.append(y_var[0])
            if d_output.has_po:
                dy_for_bwd.append(
                    xp2torch(d_output.po).reshape(
                        (d_output.po.shape[0], d_output.po.shape[2])
                    )
                )
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
                    if sgd.max_grad_norm:
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            self._model.parameters(), sgd.max_grad_norm
                        )
                    optimizer = self._optimizer

                    for params in optimizer.param_groups:
                        params["lr"] = getattr(sgd, "pytt_lr", sgd.alpha)
                    optimizer.step()
                    optimizer.zero_grad()
            return None

        self._model.eval()
        return output, backward_pytorch

    def get_model_kwargs(self, ids):
        if isinstance(ids, list):
            ids = numpy.array(ids, dtype=xp.int_)
        # Calculate "attention mask" for BERT and  XLNet, but not GPT2 (sigh)
        neg_idx = ids < 0
        ids[neg_idx] = 0
        ids = torch.as_tensor(ids, dtype=torch.int64)
        if isinstance(self._model, (pytt.BertModel, pytt.XLNetModel)):
            mask = self.ops.xp.ones(ids.shape, dtype=numpy.int_)
            mask[neg_idx] = 0
            mask = xp2torch(mask)
            segment_ids = torch.zeros_like(ids)
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": segment_ids,
            }
        else:
            return {"input_ids": ids}

    def _create_optimizer(self, sgd):
        optimizer = AdamW(
            self._model.parameters(),
            lr=sgd.alpha,
            eps=sgd.eps,
            betas=(sgd.b1, sgd.b2),
            weight_decay=getattr(sgd, "pytt_weight_decay", 0.0),
        )
        if getattr(sgd, "pytt_use_swa", False):
            optimizer = SWA(optimizer, swa_start=1, swa_freq=10, swa_lr=sgd.alpha)
        optimizer.zero_grad()
        return optimizer
