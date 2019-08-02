from thinc.api import layerize, chain, flatten_add_lengths, with_getitem
from thinc.t2v import Pooling, mean_pool
from thinc.v2v import Softmax
from thinc.neural.util import get_array_module


REGISTRY = {}


def register_model(name: str, model=None):
    """Decorator to register a model."""
    global REGISTRY
    if model is not None:
        REGISTRY[name] = model
        return model

    def do_registration(model):
        REGISTRY[name] = model
        return model

    return do_registration


def get_model_function(name: str):
    """Get a model creation function from the registry by name."""
    if name not in REGISTRY:
        names = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Model {name} not found in registry. Available names: {names}")
    return REGISTRY[name]


@register_model("fine_tune_class_vector")
def fine_tune_class_vector(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    softmax them, and then mean-pool them to produce one feature per vector.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    return chain(
        get_pytt_class_tokens,
        flatten_add_lengths,
        with_getitem(0, Softmax(nr_class, cfg["token_vector_width"])),
        Pooling(mean_pool),
    )


@layerize
def get_pytt_class_tokens(docs, drop=0.0):
    """Output a List[array], where the array is the class vector
    for each sentence in the document. To backprop, we increment the values
    in the doc._.pytt_d_last_hidden_state array.
    """
    xp = get_array_module(docs[0]._.pytt_last_hidden_state)
    outputs = []
    for doc in docs:
        wp_tensor = doc._.pytt_last_hidden_state
        class_vectors = []
        for sent in doc.sents:
            if sent._.pytt_start is not None:
                class_vectors.append(wp_tensor[sent._.pytt_start])
            else:
                class_vectors.append(xp.zeros((wp_tensor.shape[-1],), dtype="f"))
        Y = xp.vstack(class_vectors)
        outputs.append(Y)

    def backprop_pytt_class_tokens(d_outputs, sgd=None):
        for doc, dY in zip(docs, d_outputs):
            if doc._.pytt_d_last_hidden_state is None:
                xp = get_array_module(doc._.pytt_last_hidden_state)
                grads = xp.zeros(doc._.pytt_last_hidden_state.shape, dtype="f")
                doc._.pytt_d_last_hidden_state = grads
            for i, sent in enumerate(doc.sents):
                if sent._.pytt_start is not None:
                    doc._.pytt_d_last_hidden_state[sent._.pytt_start] += dY[i]
        return None

    return outputs, backprop_pytt_class_tokens


@layerize
def get_pytt_last_hidden(docs, drop=0.0):
    """Output a List[array], where the array is the last hidden vector vector
    for each document. To backprop, we increment the values
    in the doc._.pytt_d_last_hidden_state array.
    """
    outputs = [doc._.pytt_last_hidden_state for doc in docs]
    for out in outputs:
        assert out is not None

    def backprop_pytt_last_hidden(d_outputs, sgd=None):
        for doc, d_last_hidden_state in zip(docs, d_outputs):
            if doc._.pytt_d_last_hidden_state is None:
                doc._.pytt_d_last_hidden_state = d_last_hidden_state
            else:
                doc._.pytt_d_last_hidden_state += d_last_hidden_state
        return None

    return outputs, backprop_pytt_last_hidden
