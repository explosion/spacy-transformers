from typing import List, Callable
from thinc.api import Model, chain
from thinc.types import Ragged, Floats2d
from spacy.tokens import Doc

from .layers import TransformerModel, TransformerListener
from .layers import trfs2arrays, split_trf_batch
from .util import registry


@registry.architectures.register("spacy-transformers.TransformerListener.v1")
def transformer_listener_tok2vec_v1(
    pooling: Model[Ragged, Floats2d], grad_factor: float = 1.0, upstream: str = "*"
) -> Model[List[Doc], List[Floats2d]]:
    """Create a 'TransformerListener' layer, which will connect to a Transformer
    component earlier in the pipeline.

    The layer takes a list of Doc objects as input, and produces a list of
    2d arrays as output, with each array having one row per token. Most spaCy
    models expect a sublayer with this signature, making it easy to connect them
    to a transformer model via this sublayer.
    Transformer models usually operate over wordpieces, which usually don't align
    one-to-one against spaCy tokens. The layer therefore requires a reduction
    operation in order to calculate a single token vector given zero or more
    wordpiece vectors.

    pooling (Model[Ragged, Floats2d]): A reduction layer used to calculate
        the token vectors based on zero or more wordpiece vectors. If in doubt,
        mean pooling (see `thinc.layers.reduce_mean`) is usually a good choice.
    grad_factor (float): Reweight gradients from the component before passing
        them upstream. You can set this to 0 to "freeze" the transformer weights
        with respect to the component, or use it to make some components more
        significant than others. Leaving it at 1.0 is usually fine.
    upstream (str): A string to identify the 'upstream' Transformer
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the `Transformer` component. You'll almost
        never have multiple upstream Transformer components, so the wildcard
        string will almost always be fine.
    """
    listener = TransformerListener(upstream_name=upstream)
    model = chain(listener, trfs2arrays(pooling, grad_factor))
    model.set_ref("listener", listener)
    return model


@registry.architectures.register("spacy-transformers.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(
    name: str,
    get_spans,
    tokenizer_config: dict,
    pooling: Model[Ragged, Floats2d],
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    """Use a transformer as a "Tok2Vec" layer directly. This does not allow
    multiple components to share the transformer weights, and does not allow
    the transformer to set annotations into the `Doc` object, but it's a
    simpler solution if you only need the transformer within one component.

    get_spans (Callable[[List[Doc]], List[List[Span]]]): A function to extract
        spans from the batch of Doc objects. See the "TransformerModel" layer
        for details.
    tokenizer_config (dict): Settings to pass to the transformers tokenizer.
    pooling (Model[Ragged, Floats2d]): A reduction layer used to calculate
        the token vectors based on zero or more wordpiece vectors. If in doubt,
        mean pooling (see `thinc.layers.reduce_mean`) is usually a good choice.
     grad_factor (float): Reweight gradients from the component before passing
        them to the transformer. You can set this to 0 to "freeze" the transformer
        weights with respect to the component, or to make it learn more slowly.
        Leaving it at 1.0 is usually fine.
    """
    return chain(
        TransformerModel(name, get_spans, tokenizer_config),
        split_trf_batch(),
        trfs2arrays(pooling, grad_factor),
    )


# Note: when updating, also make sure to update 'replace_listener_cfg' in _util.py
@registry.architectures.register("spacy-transformers.Tok2VecTransformer.v2")
def transformer_tok2vec_v2(
    name: str,
    get_spans,
    tokenizer_config: dict,
    transformer_config: dict,
    pooling: Model[Ragged, Floats2d],
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    """Use a transformer as a "Tok2Vec" layer directly. This does not allow
    multiple components to share the transformer weights, and does not allow
    the transformer to set annotations into the `Doc` object, but it's a
    simpler solution if you only need the transformer within one component.

    get_spans (Callable[[List[Doc]], List[List[Span]]]): A function to extract
        spans from the batch of Doc objects. See the "TransformerModel" layer
        for details.
    tokenizer_config (dict): Settings to pass to the transformers tokenizer.
    transformers_config (dict): Settings to pass to the transformers forward pass
        of the transformer.
    pooling (Model[Ragged, Floats2d]): A reduction layer used to calculate
        the token vectors based on zero or more wordpiece vectors. If in doubt,
        mean pooling (see `thinc.layers.reduce_mean`) is usually a good choice.
     grad_factor (float): Reweight gradients from the component before passing
        them to the transformer. You can set this to 0 to "freeze" the transformer
        weights with respect to the component, or to make it learn more slowly.
        Leaving it at 1.0 is usually fine.
    """
    return chain(
        TransformerModel(name, get_spans, tokenizer_config, transformer_config),
        split_trf_batch(),
        trfs2arrays(pooling, grad_factor),
    )


# Note: when updating, also make sure to update 'replace_listener_cfg' in _util.py
@registry.architectures.register("spacy-transformers.Tok2VecTransformer.v3")
def transformer_tok2vec_v3(
    name: str,
    get_spans,
    tokenizer_config: dict,
    transformer_config: dict,
    pooling: Model[Ragged, Floats2d],
    grad_factor: float = 1.0,
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[List[Doc], List[Floats2d]]:
    """Use a transformer as a "Tok2Vec" layer directly. This does not allow
    multiple components to share the transformer weights, and does not allow
    the transformer to set annotations into the `Doc` object, but it's a
    simpler solution if you only need the transformer within one component.

    get_spans (Callable[[List[Doc]], List[List[Span]]]): A function to extract
        spans from the batch of Doc objects. See the "TransformerModel" layer
        for details.
    tokenizer_config (dict): Settings to pass to the transformers tokenizer.
    transformers_config (dict): Settings to pass to the transformers forward pass
        of the transformer.
    pooling (Model[Ragged, Floats2d]): A reduction layer used to calculate
        the token vectors based on zero or more wordpiece vectors. If in doubt,
        mean pooling (see `thinc.layers.reduce_mean`) is usually a good choice.
    grad_factor (float): Reweight gradients from the component before passing
        them to the transformer. You can set this to 0 to "freeze" the transformer
        weights with respect to the component, or to make it learn more slowly.
        Leaving it at 1.0 is usually fine.
    mixed_precision (bool): Enable mixed-precision. Mixed-precision replaces
        whitelisted ops to half-precision counterparts. This speeds up training
        and prediction on modern GPUs and reduces GPU memory use.
    grad_scaler_config (dict): Configuration for gradient scaling in mixed-precision
        training. Gradient scaling is enabled automatically when mixed-precision
        training is used.

        Setting `enabled` to `False` in the gradient scaling configuration disables
        gradient scaling. The `init_scale` (default: `2 ** 16`) determines the
        initial scale. `backoff_factor` (default: `0.5`) specifies the factor
        by which the scale should be reduced when gradients overflow.
        `growth_interval` (default: `2000`) configures the number of steps
        without gradient overflows after which the scale should be increased.
        Finally, `growth_factor` (default: `2.0`) determines the factor by which
        the scale should be increased when no overflows were found for
        `growth_interval` steps.
    """
    return chain(
        TransformerModel(
            name,
            get_spans,
            tokenizer_config,
            transformer_config,
            mixed_precision,
            grad_scaler_config,
        ),
        split_trf_batch(),
        trfs2arrays(pooling, grad_factor),
    )


@registry.architectures.register("spacy-transformers.TransformerModel.v1")
def create_TransformerModel_v1(
    name: str,
    get_spans: Callable,
    tokenizer_config: dict = {},
) -> Model[List[Doc], "FullTransformerBatch"]:
    model = TransformerModel(name, get_spans, tokenizer_config)
    return model


@registry.architectures.register("spacy-transformers.TransformerModel.v2")
def create_TransformerModel_v2(
    name: str,
    get_spans: Callable,
    tokenizer_config: dict = {},
    transformer_config: dict = {},
) -> Model[List[Doc], "FullTransformerBatch"]:
    model = TransformerModel(name, get_spans, tokenizer_config, transformer_config)
    return model


@registry.architectures.register("spacy-transformers.TransformerModel.v3")
def create_TransformerModel_v3(
    name: str,
    get_spans: Callable,
    tokenizer_config: dict = {},
    transformer_config: dict = {},
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[List[Doc], "FullTransformerBatch"]:
    """Pretrained transformer model that can be finetuned for downstream tasks.

    name (str): Name of the pretrained Huggingface model to use.
    get_spans (Callable[[List[Doc]], List[List[Span]]]): A function to extract
        spans from the batch of Doc objects. See the "TransformerModel" layer
        for details.
    tokenizer_config (dict): Settings to pass to the transformers tokenizer.
    transformers_config (dict): Settings to pass to the transformers forward pass
        of the transformer.
    mixed_precision (bool): Enable mixed-precision. Mixed-precision replaces
        whitelisted ops to half-precision counterparts. This speeds up training
        and prediction on modern GPUs and reduces GPU memory use.
    grad_scaler_config (dict): Configuration for gradient scaling in mixed-precision
        training. Gradient scaling is enabled automatically when mixed-precision
        training is used.

        Setting `enabled` to `False` in the gradient scaling configuration disables
        gradient scaling. The `init_scale` (default: `2 ** 16`) determines the
        initial scale. `backoff_factor` (default: `0.5`) specifies the factor
        by which the scale should be reduced when gradients overflow.
        `growth_interval` (default: `2000`) configures the number of steps
        without gradient overflows after which the scale should be increased.
        Finally, `growth_factor` (default: `2.0`) determines the factor by which
        the scale should be increased when no overflows were found for
        `growth_interval` steps.
    """
    model = TransformerModel(
        name,
        get_spans,
        tokenizer_config,
        transformer_config,
        mixed_precision,
        grad_scaler_config,
    )
    return model
