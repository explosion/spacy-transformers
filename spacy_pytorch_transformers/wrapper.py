from thinc.extra.wrappers import PyTorchWrapper, xp2torch, torch2xp
from pytorch_transformers.optimization import AdamW
import pytorch_transformers as pytt
import torch.autograd
import torch.nn.utils.clip_grad
import torch
from typing import Tuple, Callable, Any
from thinc.neural.optimizers import Optimizer
import numpy
from torchcontrib.optim import SWA

from .util import get_pytt_model
from .util import Dropout
from .activations import RaggedArray, Activations

FINE_TUNE = True
CONFIG = {"output_hidden_states": True, "output_attentions": True}


class PyTT_Wrapper(PyTorchWrapper):
    """Wrap a PyTorch-Transformers model for use in Thinc.

    The model will take as input a spacy_pytorch_transformers.util.RaggedArray
    object that will specify the input IDs and optionally the segment IDs. The
    RaggedArray is basically a tuple (ids, lengths), where ids is concatenated
    for a whole batch (this format allows the data to be contiguous even if
    the sequences are different lengths). The segment IDs should be coded as
    the different models expect them -- see
    https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py
    """

    _model: Any
    _optimizer: Any
    cfg: dict

    @classmethod
    def from_pretrained(cls, name):
        model_cls = get_pytt_model(name)
        model = model_cls.from_pretrained(name, **CONFIG)
        self = cls(name, model.config.to_dict(), model)
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
        return self.cfg.get("max_position_embeddings", 128)

    def predict(self, inputs: RaggedArray):
        self._model.eval()
        model_kwargs = self.get_model_kwargs(inputs)
        with torch.no_grad():
            if hasattr(self._optimizer, "swap_swa_sgd"):
                self._optimizer.swap_swa_sgd()
            y_var = self._model(**model_kwargs)
            if hasattr(self._optimizer, "swap_swa_sgd"):
                self._optimizer.swap_swa_sgd()
        return self.make_activations(y_var, inputs.lengths)

    def begin_update(
        self, inputs: RaggedArray, drop: Dropout = 0.0
    ) -> Tuple[Activations, Callable[..., None]]:
        if drop is None:
            # "drop is None" indicates prediction. It's one of the parts of
            # Thinc's API I'm least happy with...
            return self.predict(inputs), lambda dY, sgd=None: None
        max_original = max(inputs.lengths, default=0)
        model_kwargs = self.get_model_kwargs(inputs)
        self._model.train()
        # Prepare all the model arguments, including the attention mask
        y_var = self._model(**model_kwargs)
        output = self.make_activations(y_var, inputs.lengths)
        assert output.lh.data.shape[0] == inputs.data.shape[0], (
            output.lh.data.shape,
            inputs.data.shape,
        )

        def backward_pytorch(d_output: Activations, sgd: Optimizer = None) -> None:
            y_for_bwd = []
            dy_for_bwd = []
            if d_output.has_lh:
                assert d_output.lh.data.shape[0] == sum(d_output.lh.lengths)
                d_lh = d_output.lh.to_padded(to=max_original)
                if self.max_length and d_lh.shape[1] >= self.max_length:
                    d_lh = d_lh[:, : self.max_length]
                dy_for_bwd.append(xp2torch(d_lh))
                y_for_bwd.append(y_var[0])
            if d_output.has_po:
                dy_for_bwd.append(xp2torch(d_output.po.data))
                y_for_bwd.append(y_var[1])
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

    def make_activations(self, fields, lengths) -> Activations:
        """Create Activations from the output tuples produced by PyTorch Transformers.
        Includes converting torch tensors to xp, and handling missing values.
        """
        fields = list(fields)
        fields[0] = torch2xp(fields[0])
        fields[0] = RaggedArray.from_padded(fields[0], lengths)
        assert fields[0].data.shape[0] == sum(lengths)
        # lh: last hidden
        # po: pooler_output
        # ah: all_hidden
        # aa: all_attention
        if len(fields) != 4:
            lh = fields[0]
            po = RaggedArray.blank()
        else:
            if isinstance(fields[1], tuple):
                fields[1] = RaggedArray.blank()
            else:
                fields[1] = RaggedArray(torch2xp(fields[1]), [1] * len(lengths))
            lh, po, _, _2 = fields
        # Convert last_hidden_state to xp
        return Activations(lh, po)

    def get_model_kwargs(self, inputs):
        padded = inputs.to_padded()
        if padded.ndim == 2:
            padded = padded.reshape(padded.shape + (1,))
        if self.max_length:
            padded = padded[:, : self.max_length]
        ids = padded[:, :, 0]
        neg_idx = ids < 0
        ids[neg_idx] = 0
        ids = torch.as_tensor(ids, dtype=torch.int64)
        if padded.shape[2] == 2:
            segment_ids = padded[:, :, 1]
            segment_ids = torch.as_tensor(segment_ids, dtype=torch.int64)
        else:
            segment_ids = torch.zeros_like(ids)
        # Calculate "attention mask" for BERT and  XLNet, but not GPT2 (sigh)
        if isinstance(self._model, (pytt.BertModel, pytt.XLNetModel)):
            mask = self.ops.xp.ones(ids.shape, dtype=numpy.int_)
            mask[neg_idx] = 0
            mask = xp2torch(mask)
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": segment_ids,
            }
        else:
            return {"input_ids": ids, "token_type_ids": segment_ids}

    def _create_optimizer(self, sgd):
        optimizer = AdamW(
            self._model.parameters(),
            lr=getattr(sgd, "pytt_lr", sgd.alpha),
            eps=sgd.eps,
            betas=(sgd.b1, sgd.b2),
            weight_decay=getattr(sgd, "pytt_weight_decay", 0.0),
        )
        if getattr(sgd, "pytt_use_swa", False):
            lr = getattr(sgd, "pytt_lr", sgd.alpha)
            optimizer = SWA(optimizer, swa_start=1, swa_freq=1, swa_lr=lr)
        optimizer.zero_grad()
        return optimizer
