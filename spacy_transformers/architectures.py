from typing import List
from thinc.api import Model, chain
from thinc.types import Ragged, Floats2d
from spacy.tokens import Doc
from .layers import TransformerModel, TransformerListener
from .layers import trfs2arrays, split_trf_batch
from .util import registry


@registry.architectures.register("spacy-transformers.TransformerListener.v1")
def transformer_listener_tok2vec_v1(
    pooling: Model[Ragged, Floats2d], grad_factor: float = 1.0
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
    """
    return chain(TransformerListener("transformer"), trfs2arrays(pooling, grad_factor),)


@registry.architectures.register("spacy-transformers.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(
    name: str,
    get_spans,
    tokenizer_config,
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


registry.architectures.register(
    "spacy-transformers.TransformerModel.v1", func=TransformerModel
)
