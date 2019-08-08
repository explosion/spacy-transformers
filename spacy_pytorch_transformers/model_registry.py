from thinc.api import wrap, layerize, chain, flatten_add_lengths, with_getitem
from thinc.t2v import Pooling, mean_pool
from thinc.v2v import Softmax, Affine, Model
from thinc.neural.util import get_array_module
from spacy.tokens import Span, Doc
from typing import Tuple, Callable, List, Optional

from .wrapper import PyTT_Wrapper
from .util import Array, Dropout, Optimizer
from .util import batch_by_length, pad_batch, flatten_list, unflatten_list
from .activations import Activations as Acts


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
            with_length_batching(
                truncate_long_inputs(pytt_model, max_length), batch_by_length
            ),
        )
    )
    return model


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
        with_getitem(
            0, chain(Affine(cfg["token_vector_width"], cfg["token_vector_width"]), tanh)
        ),
        Pooling(mean_pool),
        Softmax(2, cfg["token_vector_width"]),
    )


@register_model("fine_tune_pooler_output")
def fine_tune_pooler_output(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    softmax them, and then mean-pool them to produce one feature per vector.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
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
    outputs = [doc._.pytt_pooler_output for doc in docs]

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
    outputs = []
    for sent in sents:
        wp_start = sent._.pytt_start
        wp_end = sent._.pytt_end
        if wp_start is not None and wp_end is not None:
            outputs.append(sent.doc._.pytt_word_pieces[wp_start : wp_end + 1])
        else:
            # Empty slice.
            outputs.append(sent.doc._.pytt_word_pieces[0:0])
    return outputs, None


def truncate_long_inputs(model: PyTT_Wrapper, max_len: int) -> PyTT_Wrapper:
    """Truncate inputs on the way into a model, and restore their shape on
    the way out.
    """

    def with_truncate_forward(X: Array, drop: Dropout = 0.0) -> Tuple[Acts, Callable]:
        # Dim 1 should be batch, dim 2 sequence length
        if X.shape[1] < max_len:
            return model.begin_update(X, drop=drop)
        X_short = X[:, :max_len]
        Y_short, get_dX_short = model.begin_update(X_short, drop=drop)
        outputs = Y_short.untruncate(X.shape[1])
        assert outputs.lh.shape == (X.shape[0], X.shape[1], Y_short.lh.shape[-1]), (
            X.shape,
            outputs.lh.shape,
        )

        def with_truncate_backward(dY, sgd=None):
            dY_short = dY.get_slice(slice(0, None), slice(0, max_len))
            dX_short = get_dX_short(dY_short, sgd=sgd)
            if dX_short is None:
                return None
            dX = model.ops.allocate(
                (dX_short.shape[0], dY.shape[1]) + dY_short.shape[2:]
            )
            dX[:, :max_len] = dX_short
            return dX

        return outputs, with_truncate_backward

    return wrap(with_truncate_forward, model)


def without_length_batching(model: PyTT_Wrapper) -> Model:
    """Apply padding around a model, without doing any length batching.
    
    The model should behave like PyTT_Wrapper: it should take a 2d int array
    as input, and return Activations as output. The wrapped model takes a List[Array]
    as input, and returns a List[Activations] as output.
    """

    def apply_model_padded(
        Xs: List[Array], drop: Dropout = 0.0
    ) -> Tuple[List[Acts], Callable]:
        X = pad_batch(Xs)
        A, get_dX = model.begin_update(X, drop=drop)
        As = [A.get_slice(i, slice(0, len(x))) for i, x in enumerate(Xs)]

        def backprop_batched(dAs, sgd=None):
            dA = Acts.pad_batch(dAs)
            dX = get_dX(dA, sgd=sgd)
            if dX is None:
                return None
            dXs = [dX[i, : len(x)] for i, x in enumerate(dX)]
            return dXs

        return As, backprop_batched

    return wrap(apply_model_padded, model)


def with_length_batching(model: PyTT_Wrapper, max_words: int) -> Model:
    """Wrapper that applies a model to variable-length sequences by first batching
    and padding the sequences. This allows us to group similarly-lengthed sequences
    together, making the padding less wasteful. If min_batch==1, no padding will
    be necessary.
    """
    if max_words < 1:
        return without_length_batching(model)

    def apply_model_to_batches(
        Xs: List[Array], drop: Dropout = 0.0
    ) -> Tuple[List[Acts], Callable]:
        batches: List[List[int]] = batch_by_length(Xs, max_words)
        # Initialize this, so we can place the outputs back in order.
        unbatched: List[Optional[Acts]] = [None for _ in Xs]
        backprops = []
        for indices in batches:
            X: Array = pad_batch([Xs[i] for i in indices])
            As, get_dX = model.begin_update(X, drop=drop)
            backprops.append(get_dX)
            for i, j in enumerate(indices):
                unbatched[j] = As.get_slice(i, slice(0, len(Xs[j])))
        outputs: List[Acts] = [y for y in unbatched if y is not None]
        assert len(outputs) == len(unbatched)

        def backprop_batched(d_outputs: List[Acts], sgd: Optimizer = None):
            d_inputs = [None for _ in d_outputs]
            for indices, get_dX in zip(batches, backprops):
                d_activs = Acts.pad_batch([d_outputs[i] for i in indices])
                dX = get_dX(d_activs, sgd=sgd)
                if dX is not None:
                    for i, j in enumerate(indices):
                        # As above, put things back in order, unpad.
                        d_inputs[j] = dX[i, : len(d_outputs[j])]
            assert not any(dx is None for dx in d_inputs)
            return d_inputs

        return outputs, backprop_batched

    return wrap(apply_model_to_batches, model)


def foreach_sentence(layer: Model, drop_factor: float = 1.0) -> Model:
    """Map a layer across sentences (assumes spaCy-esque .sents interface)"""
    ops = layer.ops

    Output = List[Acts]
    Backprop = Callable[[Output, Optional[Optimizer]], None]

    def sentence_fwd(docs: List[Doc], drop: Dropout = 0.0) -> Tuple[Output, Backprop]:
        sents: List[Span]
        sent_acts: List[Acts]
        bp_sent_acts: Callable[..., Optional[List[None]]]
        nested: List[List[Acts]]
        doc_sent_lengths: List[List[int]]
        doc_acts: List[Acts]

        sents = flatten_list([list(doc.sents) for doc in docs])
        sent_acts, bp_sent_acts = layer.begin_update(sents, drop=drop)
        nested = unflatten_list(sent_acts, [len(list(doc.sents)) for doc in docs])
        doc_sent_lengths = [[len(sa) for sa in doc_sa] for doc_sa in nested]
        doc_acts = [Acts.join(doc_sa) for doc_sa in nested]
        assert len(docs) == len(doc_acts)

        def sentence_bwd(d_doc_acts: Output, sgd: Optional[Optimizer] = None) -> None:
            d_nested = [
                d_doc_acts[i].split(ops, doc_sent_lengths[i])
                for i in range(len(d_doc_acts))
            ]
            d_sent_acts = flatten_list(d_nested)
            d_ids = bp_sent_acts(d_sent_acts, sgd=sgd)
            if not (d_ids is None or all(ds is None for ds in d_ids)):
                raise ValueError("Expected gradient of sentence to be None")

        return doc_acts, sentence_bwd

    model = wrap(sentence_fwd, layer)
    return model
