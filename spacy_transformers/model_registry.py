from typing import Tuple, Callable, List, Optional
from thinc.api import wrap, layerize, chain, flatten_add_lengths, with_getitem
from thinc.t2v import Pooling, mean_pool
from thinc.v2v import Softmax, Affine, Model
from thinc.neural.util import get_array_module
from spacy.tokens import Span, Doc
from spacy._ml import PrecomputableAffine, flatten
import numpy

from .wrapper import TransformersWrapper
from .util import Array, Dropout, Optimizer
from .util import batch_by_length, flatten_list, is_class_token
from .util import get_segment_ids, is_special_token, ATTRS
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
def tok2vec_per_sentence(model_name, cfg):
    max_words = cfg.get("words_per_batch", 1000)
    name = cfg["trf_model_type"]

    model = foreach_sentence(
        chain(get_word_pieces(name), with_length_batching(model_name, max_words))
    )
    return model


@register_model("softmax_tanh_class_vector")
def softmax_tanh_class_vector(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    mean-pool them, apply a tanh-activated hidden layer, and then softmax-activated
    output layer to produce one vector per document.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    width = cfg["token_vector_width"]
    return chain(
        get_class_tokens,
        flatten_add_lengths,
        with_getitem(0, chain(Affine(width, width), tanh)),
        Pooling(mean_pool),
        Softmax(nr_class, width),
    )


@register_model("softmax_class_vector")
def softmax_class_vector(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    mean-pool them, and apply a softmax-activated hidden layer to produce one
    vector per document. The gradients of the class vectors are incremented
    in the backward pass, to allow fine-tuning.
    """
    width = cfg["token_vector_width"]
    return chain(
        get_class_tokens,
        flatten_add_lengths,
        Pooling(mean_pool),
        Softmax(nr_class, width),
    )


@register_model("softmax_last_hidden")
def softmax_last_hidden(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the class-vectors from the last hidden state,
    mean-pool them, and softmax to produce one vector per document.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    width = cfg["token_vector_width"]
    return chain(
        get_last_hidden,
        flatten_add_lengths,
        Pooling(mean_pool),
        Softmax(nr_class, width),
    )


@register_model("softmax_pooler_output")
def softmax_pooler_output(nr_class, *, exclusive_classes=True, **cfg):
    """Select features from the pooler output, (if necessary) mean-pool them
    to produce one vector per item, and then softmax them.
    The gradients of the class vectors are incremented in the backward pass,
    to allow fine-tuning.
    """
    return chain(
        get_pooler_output,
        flatten_add_lengths,
        with_getitem(0, Softmax(nr_class, cfg["token_vector_width"])),
        Pooling(mean_pool),
    )

@register_model("tensor_affine_tok2vec")
def tensor_affine_tok2vec(output_size, tensor_size, **cfg):
    return chain(
        get_tensors,
        flatten,
        Affine(output_size, tensor_size)
    )


@register_model("precomputable_maxout")
def precompute_hiddens(nO, nI, nF, nP, **cfg):
    return PrecomputableAffine(hidden_width, nF=nr_feature,
        nI=token_vector_width, nP=parser_maxout_pieces)
 

@register_model("affine_output")
def affine_output(nO, nI, drop_factor, **cfg):
    return Affine(nO, nI, drop_factor=drop_factor)
 

@layerize
def get_tensors(docs, drop=0.0):
    """Output a List[array], where the array is the tensor of each document."""
    tensors = [doc.tensor for doc in docs]

    def backprop_tensors(d_tensors, sgd=None):
        for doc, d_t in zip(docs, d_tensor):
            # Count how often each word-piece token is represented. This allows
            # a weighted sum, so that we can make sure doc.tensor.sum()
            # equals wp_tensor.sum(). Do this with sensitivity to boundary tokens
            wp_rows, align_sizes = _get_boundary_sensitive_alignment(doc)
            d_lh = _get_or_set_d_last_hidden_state(doc)
            for i, word_piece_slice in enumerate(wp_rows):
                for j in word_piece_slice:
                    d_lh[j] += d_tensor[i]
            xp = get_array_module(d_lh)
            d_lh /= xp.array(align_sizes, dtype="f").reshape(-1, 1)
            return None

    return tensors, backprop_tensors


def _get_or_set_d_last_hidden_state(doc):
    xp = get_array_model(doc._.get(ATTRS.last_hidden_state))
    if doc._.get(ATTRS.d_last_hidden_state).size == 0:
        shape = doc._.get(ATTRS.last_hidden_state).shape
        dtype = doc._.get(ATTRS.last_hidden_state).dtype
        doc._.set(ATTRS.d_last_hidden_state, xp.zeros(shape, dtype=dtype))
    return doc._.get(ATTRS.d_last_hidden_state)


@layerize
def get_class_tokens(docs, drop=0.0):
    """Output a List[array], where the array is the class vector
    for each sentence in the document. To backprop, we increment the values
    in the Doc's d_last_hidden_state array.
    """
    xp = get_array_module(docs[0]._.get(ATTRS.last_hidden_state))
    outputs = []
    doc_class_tokens = []
    for doc in docs:
        class_tokens = []
        for i, wp in enumerate(doc._.get(ATTRS.word_pieces_)):
            if is_class_token(wp):
                class_tokens.append(i)
        doc_class_tokens.append(xp.array(class_tokens, dtype="i"))
        wp_tensor = doc._.get(ATTRS.last_hidden_state)
        outputs.append(wp_tensor[doc_class_tokens[-1]])

    def backprop_class_tokens(d_outputs, sgd=None):
        for doc, class_tokens, dY in zip(docs, doc_class_tokens, d_outputs):
            if doc._.get(ATTRS.d_last_hidden_state).size == 0:
                xp = get_array_module(doc._.get(ATTRS.last_hidden_state))
                grads = xp.zeros(doc._.get(ATTRS.last_hidden_state).shape, dtype="f")
                doc._.set(ATTRS.d_last_hidden_state, grads)
            doc._.get(ATTRS.d_last_hidden_state)[class_tokens] += dY
        return None

    return outputs, backprop_class_tokens


get_class_tokens.name = "get_class_tokens"


@layerize
def get_pooler_output(docs, drop=0.0):
    """Output a List[array], where the array is the class vector
    for each sentence in the document. To backprop, we increment the values
    in the Doc's d_last_hidden_state array.
    """
    for doc in docs:
        if doc._.get(ATTRS.pooler_output) is None:
            raise ValueError(
                "Pooler output unset. Perhaps you're using the wrong architecture? "
                "The BERT model provides a pooler output, but XLNet doesn't. "
                "You might need to set 'architecture': 'softmax_class_vector' "
                "instead."
            )
    outputs = [doc._.get(ATTRS.pooler_output) for doc in docs]

    def backprop_pooler_output(d_outputs, sgd=None):
        for doc, dY in zip(docs, d_outputs):
            if doc._.get(ATTRS.d_pooler_output).size == 0:
                xp = get_array_module(doc._.get(ATTRS.pooler_output))
                grads = xp.zeros(doc._.get(ATTRS.pooler_output).shape, dtype="f")
                doc._.set(ATTRS.d_pooler_output, grads)
            doc._.set(ATTRS.d_pooler_output, doc._.get(ATTRS.d_pooler_output) + dY)
        return None

    return outputs, backprop_pooler_output


get_pooler_output.name = "get_pooler_output"


@layerize
def get_last_hidden(docs, drop=0.0):
    """Output a List[array], where the array is the last hidden vector vector
    for each document. To backprop, we increment the values
    in the Doc's d_last_hidden_state array.
    """
    outputs = [doc._.get(ATTRS.last_hidden_state) for doc in docs]
    for out in outputs:
        assert out is not None
        assert out.size != 0

    def backprop_last_hidden(d_outputs, sgd=None):
        for doc, d_lh in zip(docs, d_outputs):
            xp = get_array_module(d_lh)
            shape = d_lh.shape
            dtype = d_lh.dtype
            if doc._.get(ATTRS.d_last_hidden_state).size == 0:
                doc._.set(ATTRS.d_last_hidden_state, xp.zeros(shape, dtype=dtype))
            doc._.set(
                ATTRS.d_last_hidden_state, doc._.get(ATTRS.d_last_hidden_state) + d_lh
            )
        return None

    return outputs, backprop_last_hidden


get_last_hidden.name = "get_last_hidden"


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


def get_word_pieces(transformers_name):
    def get_features_forward(sents, drop=0.0):
        assert isinstance(sents[0], Span)
        ids = []
        segment_ids = []
        lengths = []
        for sent in sents:
            wordpieces = sent._.get(ATTRS.word_pieces)
            # This is a bit convoluted, but we need the lengths without any
            # separator tokens or class tokens. `segments` gives Span objects.
            seg_lengths = [
                len(
                    [
                        w
                        for w in seg._.get(ATTRS.word_pieces_)
                        if not is_special_token(w)
                    ]
                )
                for seg in sent._.get(ATTRS.segments)
            ]
            if wordpieces:
                ids.extend(wordpieces)
                lengths.append(len(wordpieces))
                sent_seg_ids = get_segment_ids(transformers_name, *seg_lengths)
                segment_ids.extend(sent_seg_ids)
                assert len(wordpieces) == len(sent_seg_ids), (
                    sent._.get(ATTRS.word_pieces_),
                    seg_lengths,
                    len(wordpieces),
                    len(sent_seg_ids),
                )
            else:
                lengths.append(0)
        assert len(ids) == len(segment_ids), (len(ids), len(segment_ids))
        features = numpy.array(list(zip(ids, segment_ids)), dtype=numpy.int_)
        assert features.shape[0] == sum(lengths), (features.shape, sum(lengths))
        return RaggedArray(features, lengths), None

    return layerize(get_features_forward, name="get_features_forward")


def with_length_batching(
    model: TransformersWrapper, max_words: int
) -> TransformersWrapper:
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
            if outputs.has_po and po_rows.size:
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
        if not all(doc.is_sentenced for doc in docs):
            return layer.begin_update([d[:] for d in docs], drop=drop)
        sents = flatten_list([list(doc.sents) for doc in docs])
        words_per_doc = [len(d._.get(ATTRS.word_pieces)) for d in docs]
        words_per_sent = [len(s._.get(ATTRS.word_pieces)) for s in sents]
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

    return wrap(sentence_fwd, layer)
