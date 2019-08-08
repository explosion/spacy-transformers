from typing import Tuple, Callable, List, Optional
from thinc.api import wrap, layerize, chain, flatten_add_lengths, with_getitem
from thinc.t2v import Pooling, mean_pool
from thinc.v2v import Softmax, Affine, Model
from thinc.neural.util import get_array_module
from spacy.tokens import Span, Doc
import numpy

from .wrapper import PyTT_Wrapper
from .util import Array, Dropout, Optimizer
from .util import batch_by_length, pad_batch, flatten_list, unflatten_list
from .activations import Activations as Acts
from .activations import RaggedArray


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


@register_model("tok2vec_per_sentence")
def tok2vec_per_sentence(pytt_model, cfg):
    batch_by_length = cfg.get("words_per_batch", 5000)
    max_length = cfg.get("max_length", 512)

    model = foreach_sentence(
        chain(
            get_word_pieces,
            without_length_batching(
                truncate_long_inputs(pytt_model, max_length)
            ),
        )
    )
    return model


@register_model("softmax_class_vector")
def softmax_class_vector(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    mean-pool them, and softmax to produce one vector per document.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    return chain(
        get_pytt_class_tokens,
        flatten_add_lengths,
        with_getitem(
            0, chain(Affine(cfg["token_vector_width"], cfg["token_vector_width"]), tanh)
        ),
        Pooling(mean_pool),
        Softmax(2, cfg["token_vector_width"]),
    )


@register_model("softmax_pooler_output")
def softmax_pooler_output(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the pooler output, (if necessary) mean-pool them,
    mean-pool them to produce one vector per softmax them, and then mean-pool
    them to produce one feature per vector. The gradients of the class vectors
    are incremented in the backward pass, to allow fine-tuning.
    """
    return chain(
        get_pytt_pooler_output,
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
            if doc._.pytt_d_last_hidden_state.size == 0:
                xp = get_array_module(doc._.pytt_last_hidden_state)
                grads = xp.zeros(doc._.pytt_last_hidden_state.shape, dtype="f")
                doc._.pytt_d_last_hidden_state = grads
            for i, sent in enumerate(doc.sents):
                if sent._.pytt_start is not None:
                    doc._.pytt_d_last_hidden_state[sent._.pytt_start] += dY[i]
        return None

    return outputs, backprop_pytt_class_tokens


@layerize
def get_pytt_pooler_output(docs, drop=0.0):
    """Output a List[array], where the array is the class vector
    for each sentence in the document. To backprop, we increment the values
    in the doc._.pytt_d_last_hidden_state array.
    """
    outputs = [doc._.pytt_pooler_output.sum(axis=0) for doc in docs]

    def backprop_pytt_pooler_output(d_outputs, sgd=None):
        for doc, dY in zip(docs, d_outputs):
            if doc._.pytt_d_pooler_output.size == 0:
                xp = get_array_module(doc._.pytt_pooler_output)
                grads = xp.zeros(doc._.pytt_pooler_output.shape, dtype="f")
                doc._.pytt_d_pooler_output = grads
            doc._.pytt_d_pooler_output += dY
        return None

    return outputs, backprop_pytt_pooler_output


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


@layerize
def softmax(X, drop=0.0):
    ops = Model.ops
    Y = ops.softmax(X)

    def backprop_softmax(dY, sgd=None):
        dX = ops.backprop_softmax(Y, dY)
        return dX

    return Y, backprop_softmax


@layerize
def tanh(X, drop=0.0):
    xp = get_array_module(X)
    Y = xp.tanh(X)

    def backprop_tanh(dY, sgd=None):
        one = Y.dtype.type(1)
        dX = dY * (one - Y * Y)
        return dX

    return Y, backprop_tanh


@layerize
def get_word_pieces(sents, drop=0.0):
    assert isinstance(sents[0], Span)
    ids = []
    lengths = []
    for sent in sents:
        wp_start = sent._.pytt_start
        wp_end = sent._.pytt_end
        if wp_start is not None and wp_end is not None:
            ids.extend(sent.doc._.pytt_word_pieces[wp_start : wp_end + 1])
            lengths.append((wp_end+1)-wp_start)
        else:
            lengths.append(0)
    return RaggedArray(numpy.array(ids, dtype=numpy.int_), lengths), None


def truncate_long_inputs(model: PyTT_Wrapper, max_len: int) -> PyTT_Wrapper:
    """Truncate inputs on the way into a model, and restore their shape on
    the way out.
    """

    def with_truncate_forward(inputs: RaggedArray, drop: Dropout = 0.0) -> Tuple[Acts, Callable]:
        return model.begin_update(inputs, drop=drop)

    return wrap(with_truncate_forward, model)


def without_length_batching(model: PyTT_Wrapper) -> PyTT_Wrapper:

    def apply_model_unpadded(inputs: RaggedArray, drop=0.) -> Tuple[Acts, Callable]:
        assert isinstance(inputs, RaggedArray)
        return model.begin_update(inputs, drop=drop)

    return wrap(apply_model_unpadded, model)


def foreach_sentence(layer: Model, drop_factor: float = 1.0) -> Model:
    """Map a layer across sentences (assumes spaCy-esque .sents interface)"""
    ops = layer.ops

    def sentence_fwd(docs: List[Doc], drop: Dropout = 0.0) -> Tuple[Acts, Callable]:
        sents = flatten_list([list(doc.sents) for doc in docs])
        words_per_doc = [len(d._.pytt_word_pieces) for d in docs]
        words_per_sent = [len(s._.pytt_word_pieces) for s in sents]
        sents_per_doc = [len(list(d.sents)) for d in docs]
        acts, bp_acts = layer.begin_update(sents, drop=drop)
        # To go from "per sentence" activations to "per doc" activations, we
        # just have to tell it where the sequences end.
        acts.lh.lengths = words_per_doc
        acts.po.lengths = sents_per_doc

        def sentence_bwd(d_acts: Acts, sgd: Optional[Optimizer] = None) -> None:
            # Translate back to the per-sentence activations
            d_acts.lh.lengths = words_per_sent
            d_acts.po.lengths = [1 for _ in words_per_sent]
            d_ids = bp_acts(d_acts, sgd=sgd)
            if not (d_ids is None or all(ds is None for ds in d_ids)):
                raise ValueError("Expected gradient of sentence to be None")
            return d_ids

        return acts, sentence_bwd

    model = wrap(sentence_fwd, layer)
    return model
