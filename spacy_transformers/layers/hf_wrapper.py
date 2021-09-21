from typing import Callable, Optional, Any
from thinc.layers.pytorchwrapper import forward as pt_forward
from thinc.layers.pytorchwrapper import convert_pytorch_default_inputs, convert_pytorch_default_outputs

from thinc.api import registry, Model

from .hf_shim import HFShim


@registry.layers("HFWrapper.v1")
def HFWrapper(
    hf_model: "HFObjects",
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[Any, Any]:
    """Wrap a PyTorch HF model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch. See examples/wrap_pytorch.py

    Your PyTorch model's forward method can take arbitrary args and kwargs,
    but must return either a single tensor as output or a tuple. You may find the
    PyTorch register_forward_hook helpful if you need to adapt the output.

    The convert functions are used to map inputs and outputs to and from your
    PyTorch model. Each function should return the converted output, and a callback
    to use during the backward pass. So:

        Xtorch, get_dX = convert_inputs(X)
        Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
        Y, get_dYtorch = convert_outputs(Ytorch)

    To allow maximum flexibility, the PyTorchShim expects ArgsKwargs objects
    on the way into the forward and backward passed. The ArgsKwargs objects
    will be passed straight into the model in the forward pass, and straight
    into `torch.autograd.backward` during the backward pass.
    """
    if convert_inputs is None:
        convert_inputs = convert_pytorch_default_inputs
    if convert_outputs is None:
        convert_outputs = convert_pytorch_default_outputs

    return Model(
        "hf-pytorch",
        pt_forward,
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
        shims=[
            HFShim(
                hf_model,
                mixed_precision=mixed_precision,
                grad_scaler_config=grad_scaler_config,
            )
        ],
        dims={"nI": None, "nO": None},
    )
