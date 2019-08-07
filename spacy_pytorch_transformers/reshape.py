"""Thinc layers for data manipulation."""
from .activations import Activations as Acts


def truncate_long_inputs(model: PyTT_Wrapper, max_len: int) -> PyTT_Wrapper:
    """Truncate inputs on the way into a model, and restore their shape on
    the way out.
    """

    def with_truncate_forward(
        X: Array, drop: Dropout = 0.0
    ) -> Tuple[Acts, Callable]:
        # Dim 1 should be batch, dim 2 sequence length
        if X.shape[1] < max_len:
            return model.begin_update(X, drop=drop)
        X_short = X[:, :max_len]
        Y_short, get_dX_short = model.begin_update(X_short, drop=drop)
        outputs = Y_short.untruncate(X.shape[1])
        assert outputs.lh.shape == (X.shape[0], X.shape[1], Y_short.lh.shape[-1]), (X.shape, outputs.lh.shape)

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
    def apply_model_padded(Xs: List[Array], drop: Dropout = 0.0) -> Tuple[List[Acts], Callable]:
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

    def apply_model_to_batches(Xs: List[Array], drop: Dropout = 0.0) -> Tuple[List[Acts], Callable]:
        batches: List[List[int]] = batch_by_length(Xs, max_words)
        # Initialize this, so we can place the outputs back in order.
        unbatched: List[Optional[Acts]] = [None for _ in inputs]
        backprops = []
        for indices in batches:
            X: Array = pad_batch([inputs[i] for i in indices])
            As, get_dX = model.begin_update(X, drop=drop)
            backprops.append(get_dX)
            for i, j in enumerate(indices):
                unbatched[j] = As.get_slice(i, slice(0, len(inputs[j])))
        outputs: List[Acts] = [y for y in unbatched if y is not None]
        assert len(outputs) == len(unbatched)

        def backprop_batched(d_outputs: List[Acts], sgd: Optimizer = None):
            d_inputs = [None for _ in inputs]
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
