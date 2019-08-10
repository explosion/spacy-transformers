from typing import Tuple, Callable, List, Optional
from thinc.api import wrap, layerize, chain, flatten_add_lengths, with_getitem
from thinc.t2v import Pooling, mean_pool
from thinc.v2v import Softmax, Affine, Model
from thinc.neural.util import get_array_module
from spacy.tokens import Span, Doc
import numpy

from .wrapper import PyTT_Wrapper
from .util import Array, Dropout, Optimizer
from .util import batch_by_length, flatten_list
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
    max_words = cfg.get("words_per_batch", 2000)

    model = foreach_sentence(
        chain(get_word_pieces, with_length_batching(pytt_model, max_words))
    )
    return model


@register_model("softmax_tanh_class_vector")
def softmax_tanh_class_vector(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    mean-pool them, and softmax to produce one vector per document.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    width = cfg["token_vector_width"]
    return chain(
        get_pytt_class_tokens,
        flatten_add_lengths,
        with_getitem(0, chain(Affine(width, width), tanh)),
        Pooling(mean_pool),
        Softmax(2, width),
    )


@register_model("softmax_class_vector")
def softmax_class_vector(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    mean-pool them, and softmax to produce one vector per document.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    width = cfg["token_vector_width"]
    return chain(
        get_pytt_class_tokens,
        flatten_add_lengths,
        Pooling(mean_pool),
        Softmax(2, width),
    )


@register_model("softmax_pooler_output")
def softmax_pooler_output(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the pooler output, (if necessary) mean-pool them
    to produce one vector per item, and then softmax them.
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
            nr_word_pieces = len(doc._.pytt_word_pieces)
            assert doc._.pytt_d_last_hidden_state.shape[0] == nr_word_pieces
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
    ids = []
    lengths = []
    for sent in sents:
        wp_start = sent._.pytt_start
        wp_end = sent._.pytt_end
        if wp_start is not None and wp_end is not None:
            ids.extend(sent.doc._.pytt_word_pieces[wp_start : wp_end + 1])
            lengths.append((wp_end + 1) - wp_start)
        else:
            lengths.append(0)
    return RaggedArray(numpy.array(ids, dtype=numpy.int_), lengths), None


def with_length_batching(model: PyTT_Wrapper, max_words: int) -> PyTT_Wrapper:
    ops = model.ops

    def apply_model_to_batches(
        inputs: RaggedArray, drop: Dropout = 0.0
    ) -> Tuple[Acts, Callable]:
        if max_words == 0 or inputs.data.shape[0] < max_words:
            return model.begin_update(inputs, drop=drop)
        Xs: List[Array] = ops.unflatten(inputs.data, inputs.lengths)
        outputs = None
        backprops = []
        index2rows = {}
        start = 0
        # Map each index to the slice of rows in the flattened data it refers to.
        for i, length in enumerate(inputs.lengths):
            index2rows[i] = [start + j for j in range(length)]
            start += length
        total_rows = sum(inputs.lengths)
        for indices in batch_by_length(Xs, max_words):
            X: Array = inputs.xp.concatenate([Xs[i] for i in indices])
            lengths = [inputs.lengths[i] for i in indices]
            Y, get_dX = model.begin_update(RaggedArray(X, lengths), drop=drop)
            if outputs is None:
                lh_shape = (total_rows, Y.lh.data.shape[-1])
                po_shape = (len(inputs.lengths), Y.po.data.shape[-1])
                outputs = Acts(
                    RaggedArray(Y.lh.xp.zeros(lh_shape, dtype="f"), inputs.lengths),
                    RaggedArray(
                        Y.po.xp.zeros(po_shape, dtype="f"), [1 for _ in inputs.lengths]
                    ),
                )
            lh_rows = []
            po_rows = []
            for index in indices:
                lh_rows.extend(index2rows[index])
                po_rows.append(index)
            lh_rows = outputs.xp.array(lh_rows, dtype="i")
            po_rows = outputs.xp.array(po_rows, dtype="i")
            outputs.lh.data[lh_rows] = Y.lh.data
            outputs.po.data[po_rows] = Y.po.data
            backprops.append((get_dX, lh_rows, po_rows, lengths))

        def backprop_batched(d_outputs: Acts, sgd: Optimizer = None):
            for get_dX, lh_rows, po_rows, lengths in backprops:
                if d_outputs.has_lh:
                    d_lh = d_outputs.lh.data[lh_rows]
                    lh_lengths = lengths
                else:
                    d_lh = d_outputs.lh.data
                    lh_lengths = []
                if d_outputs.has_po:
                    d_po = d_outputs.po.data[po_rows]
                    po_lengths = [1 for _ in lengths]
                else:
                    d_po = d_outputs.po.data
                    po_lengths = []
                dY = Acts(RaggedArray(d_lh, lh_lengths), RaggedArray(d_po, po_lengths))
                dX = get_dX(dY, sgd=sgd)
                assert dX is None
            return None

        return outputs, backprop_batched

    return wrap(apply_model_to_batches, model)


def foreach_sentence(layer: Model, drop_factor: float = 1.0) -> Model:
    """Map a layer across sentences (assumes spaCy-esque .sents interface)"""

    def sentence_fwd(docs: List[Doc], drop: Dropout = 0.0) -> Tuple[Acts, Callable]:
        sents = flatten_list([list(doc.sents) for doc in docs])
        words_per_doc = [len(d._.pytt_word_pieces) for d in docs]
        words_per_sent = [len(s._.pytt_word_pieces) for s in sents]
        sents_per_doc = [len(list(d.sents)) for d in docs]
        assert sum(words_per_doc) == sum(words_per_sent)
        acts, bp_acts = layer.begin_update(sents, drop=drop)
        # To go from "per sentence" activations to "per doc" activations, we
        # just have to tell it where the sequences end.
        acts.lh.lengths = words_per_doc
        acts.po.lengths = sents_per_doc

        def sentence_bwd(d_acts: Acts, sgd: Optional[Optimizer] = None) -> None:
            assert isinstance(d_acts, Acts)
            # Translate back to the per-sentence activations
            if d_acts.has_lh:
                assert d_acts.lh.data.shape[0] == sum(d_acts.lh.lengths)
                assert d_acts.lh.lengths == words_per_doc
            d_acts.lh.lengths = words_per_sent
            d_acts.po.lengths = [1 for _ in words_per_sent]
            d_ids = bp_acts(d_acts, sgd=sgd)
            if not (d_ids is None or all(ds is None for ds in d_ids)):
                raise ValueError("Expected gradient of sentence to be None")
            return d_ids

        return acts, sentence_bwd

    model = wrap(sentence_fwd, layer)
    return model
