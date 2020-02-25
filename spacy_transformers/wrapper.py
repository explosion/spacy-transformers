from thinc.extra.wrappers import PyTorchWrapper, xp2torch, torch2xp
from transformers.optimization import AdamW
import transformers
import torch.autograd
import torch.nn.utils.clip_grad
import torch
from typing import Tuple, Callable, Any
from thinc.neural.optimizers import Optimizer
import numpy
import contextlib
from thinc.compat import BytesIO

from .util import get_model, Dropout
from .activations import RaggedArray, Activations


FINE_TUNE = True
CONFIG = {"output_hidden_states": True, "output_attentions": True}


class TransformersWrapper(PyTorchWrapper):
    """Wrap a Transformers model for use in Thinc.

    The model will take as input a spacy_transformers.util.RaggedArray
    object that will specify the input IDs and optionally the segment IDs. The
    RaggedArray is basically a tuple (ids, lengths), where ids is concatenated
    for a whole batch (this format allows the data to be contiguous even if
    the sequences are different lengths). The segment IDs should be coded as
    the different models expect them -- see
    https://github.com/huggingface/transformers/blob/master/examples/utils_glue.py
    """

    _model: Any
    _optimizer: Any
    cfg: dict

    @classmethod
    def from_pretrained(cls, name):
        model_cls = get_model(name)
        model = model_cls.from_pretrained(name, **CONFIG)
        self = cls(name, model.config.to_dict(), model)
        self.cfg.update(self.transformers_model.config.to_dict())
        return self

    def __init__(self, name, config, model):
        PyTorchWrapper.__init__(self, model)
        self.cfg = dict(config)

    @property
    def nO(self):
        if "hidden_size" in self.cfg:
            # BERT
            return self.cfg["hidden_size"]
        elif "hidden_dim" in self.cfg:
            # DistilBERT
            return self.cfg["hidden_dim"] // 4
        elif "n_embd" in self.cfg:
            # GPT2
            return self.cfg["n_embd"]
        elif "d_model" in self.cfg:
            # XLNet
            return self.cfg["d_model"]
        elif hasattr(self.transformers_model, "dim"):
            # XLM
            return self.transformers_model.dim
        else:
            keys = ", ".join(self.cfg.keys())
            raise ValueError(f"Unexpected config. Keys: {keys}")

    @property
    def transformers_model(self):
        return self._model

    @property
    def max_length(self):
        # `n_positions` in GPT2 config
        return self.cfg.get("max_position_embeddings", self.cfg.get("n_positions", 128))

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
                    for group in optimizer.param_groups:
                        group["lr"] = getattr(sgd, "trf_lr", sgd.alpha)
                    optimizer.step()
                    optimizer.zero_grad()
                    self._update_pytorch_averages(sgd)
            return None

        self._model.eval()
        return output, backward_pytorch

    @contextlib.contextmanager
    def use_params(self, params):
        key_prefix = f"pytorch_{self.id}_"
        state_dict = {}
        for k, v in params.items():
            if hasattr(k, "startswith") and k.startswith(key_prefix):
                state_dict[k.replace(key_prefix, "")] = xp2torch(v)
        if state_dict:
            backup = {k: v.clone() for k, v in self._model.state_dict().items()}
            self._model.load_state_dict(state_dict)
            yield
            self._model.load_state_dict(backup)
        else:
            yield

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
        padded = inputs.to_padded(value=-1)
        if padded.ndim == 2:
            padded = padded.reshape(padded.shape + (1,))
        ids = padded[:, :, 0]
        neg_idx = ids < 0
        ids[neg_idx] = 0
        ids = torch.as_tensor(ids, dtype=torch.int64)
        if padded.shape[2] == 2:
            segment_ids = padded[:, :, 1]
            numpy.place(segment_ids, segment_ids<0, 0)
            segment_ids = torch.as_tensor(segment_ids, dtype=torch.int64)
        else:
            segment_ids = torch.zeros_like(ids)
        # Calculate "attention mask" for BERT and  XLNet, but not GPT2 (sigh)
        if isinstance(self._model, (transformers.BertModel, transformers.XLNetModel)):
            mask = self.ops.xp.ones(ids.shape, dtype=numpy.int_)
            mask[neg_idx] = 0
            mask = xp2torch(mask)
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": segment_ids,
            }
        elif isinstance(self._model, (transformers.DistilBertModel)):
            # Mask, but no token type IDs for DistilBert (sigh again...)
            mask = self.ops.xp.ones(ids.shape, dtype=numpy.int_)
            mask[neg_idx] = 0
            mask = xp2torch(mask)
            return {"input_ids": ids, "attention_mask": mask}
        else:
            return {"input_ids": ids, "token_type_ids": segment_ids}

    def _create_optimizer(self, sgd):
        optimizer = AdamW(
            self._model.parameters(),
            lr=getattr(sgd, "trf_lr", sgd.alpha),
            eps=sgd.eps,
            betas=(sgd.b1, sgd.b2),
            weight_decay=getattr(sgd, "trf_weight_decay", 0.0),
        )
        optimizer.zero_grad()
        return optimizer

    def _update_pytorch_averages(self, sgd, *, init_steps=1):
        if sgd.averages is None:
            return
        # Collect parameters if we don't have them
        for name, param in self._model.state_dict().items():
            key = f"pytorch_{self.id}_{name}"
            sgd.nr_update[key] += 1
            xp_param = torch2xp(param)
            if key in sgd.averages:
                self.ops.update_averages(
                    sgd.averages[key], xp_param, sgd.nr_update[key]
                )
            else:
                sgd.averages[key] = xp_param.copy()
                sgd.nr_update[key] = init_steps

    def to_disk(self, path):
        torch.save(self._model.state_dict(), str(path))

    def from_disk(self, path):
        if self.ops.device == "cpu":
            map_location = "cpu"
        else:
            map_location = "cuda:0"
        self._model.load_state_dict(torch.load(path, map_location=map_location))
        self._model.to(map_location)

    def to_bytes(self):
        filelike = BytesIO()
        torch.save(self._model.state_dict(), filelike)
        filelike.seek(0)
        return filelike.getvalue()

    def from_bytes(self, data):
        filelike = BytesIO(data)
        filelike.seek(0)
        if self.ops.device == "cpu":
            map_location = "cpu"
        else:
            map_location = "cuda:0"
        self._model.load_state_dict(torch.load(filelike, map_location=map_location))
        self._model.to(map_location)
