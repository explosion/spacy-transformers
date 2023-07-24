from typing import Any, Dict
from io import BytesIO
from pathlib import Path
import srsly
import torch
import warnings
from thinc.api import get_torch_default_device
from spacy.util import SimpleFrozenDict

from ..data_classes import HFObjects
from ..util import make_tempdir

from thinc.api import PyTorchGradScaler, PyTorchShim

from transformers import AutoModel, AutoConfig, AutoTokenizer


class HFShim(PyTorchShim):
    """Interface between a HF Pytorch model and a Thinc Model."""

    def __init__(
        self,
        model: HFObjects,
        config=None,
        optimizer: Any = None,
        mixed_precision: bool = False,
        grad_scaler_config: dict = {},
        config_cls=AutoConfig,
        model_cls=AutoModel,
        tokenizer_cls=AutoTokenizer,
    ):
        self._hfmodel = model
        self.config_cls = config_cls
        self.model_cls = model_cls
        self.tokenizer_cls = tokenizer_cls

        # Enable gradient scaling when mixed precision is enabled and gradient
        # scaling is not explicitly disabled in the configuration.
        if "enabled" not in grad_scaler_config:
            grad_scaler_config["enabled"] = mixed_precision

        super().__init__(
            model.transformer,
            config,
            optimizer,
            mixed_precision,
            grad_scaler=PyTorchGradScaler(**grad_scaler_config),
        )

    def to_bytes(self):
        config = {}
        tok_dict = {}
        weights_bytes = {}
        tok_cfg = {}
        trf_cfg = {}
        hf_model = self._hfmodel
        if hf_model.transformer is not None:
            tok_dict = {}
            config = hf_model.transformer.config.to_dict()
            tokenizer = hf_model.tokenizer
            with make_tempdir() as temp_dir:
                if hasattr(tokenizer, "vocab_file"):
                    vocab_file_name = tokenizer.vocab_files_names["vocab_file"]
                    vocab_file_path = str((temp_dir / vocab_file_name).absolute())
                    with open(vocab_file_path, "wb") as fileh:
                        fileh.write(hf_model.vocab_file_contents)
                    tokenizer.vocab_file = vocab_file_path
                tok_dict["kwargs"] = {"use_fast": tokenizer.is_fast}
                tokenizer.save_pretrained(str(temp_dir.absolute()))
                for x in temp_dir.glob("**/*"):
                    if x.is_file():
                        tok_dict[x.name] = x.read_bytes()
            filelike = BytesIO()
            torch.save(self._model.state_dict(), filelike)
            filelike.seek(0)
            weights_bytes = filelike.getvalue()
        else:
            tok_cfg = hf_model._init_tokenizer_config
            trf_cfg = hf_model._init_transformer_config
        msg = {
            "config": config,
            "state": weights_bytes,
            "tokenizer": tok_dict,
            "_init_tokenizer_config": tok_cfg,
            "_init_transformer_config": trf_cfg,
        }
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data):
        msg = srsly.msgpack_loads(bytes_data)
        config_dict = msg["config"]
        tok_dict = msg["tokenizer"]
        if config_dict:
            with make_tempdir() as temp_dir:
                config_file = temp_dir / "config.json"
                srsly.write_json(config_file, config_dict)
                config = self.config_cls.from_pretrained(config_file)
                tok_kwargs = tok_dict.pop("kwargs", {})
                for x, x_bytes in tok_dict.items():
                    Path(temp_dir / x).write_bytes(x_bytes)
                tokenizer = self.tokenizer_cls.from_pretrained(
                    str(temp_dir.absolute()), **tok_kwargs
                )
                vocab_file_contents = None
                if hasattr(tokenizer, "vocab_file"):
                    vocab_file_name = tokenizer.vocab_files_names["vocab_file"]
                    vocab_file_path = str((temp_dir / vocab_file_name).absolute())
                    with open(vocab_file_path, "rb") as fileh:
                        vocab_file_contents = fileh.read()

            transformer = self.model_cls.from_config(config)
            self._hfmodel = HFObjects(
                tokenizer,
                transformer,
                vocab_file_contents,
                SimpleFrozenDict(),
                SimpleFrozenDict(),
            )
            self._model = transformer
            filelike = BytesIO(msg["state"])
            filelike.seek(0)
            device = get_torch_default_device()
            try:
                self._model.load_state_dict(torch.load(filelike, map_location=device))
            except RuntimeError as ex:
                warn_msg = (
                    "Error loading saved torch model. If the error is related "
                    "to unexpected key(s) in state_dict, a possible workaround "
                    "is to load this model with 'transformers<4.31'. "
                    "Alternatively, download a newer compatible model or "
                    "retrain your custom model with the current "
                    "transformers and spacy-transformers versions. For more "
                    "details and available updates, run: python -m spacy "
                    "validate"
                )
                warnings.warn(warn_msg)
                raise ex
            self._model.to(device)
        else:
            self._hfmodel = HFObjects(
                None,
                None,
                None,
                msg["_init_tokenizer_config"],
                msg["_init_transformer_config"],
            )
        return self
