from spacy.pipeline import Pipe
from thinc.v2v import Model
from thinc.api import chain, layerize, wrap
from thinc.neural.ops import get_array_module
from spacy.util import minibatch
from spacy.tokens import Span

from .wrapper import PyTT_Wrapper
from .util import batch_by_length, pad_batch, flatten_list, unflatten_list, Activations


class PyTT_TokenVectorEncoder(Pipe):
    """spaCy pipeline component to use PyTorch-Transformers models.

    The component assigns the output of the transformer to the `doc._.pytt_outputs`
    extension attribute. We also calculate an alignment between the word-piece
    tokens and the spaCy tokenization, so that we can use the last hidden states
    to set the doc.tensor attribute. When multiple word-piece tokens align to
    the same spaCy token, the spaCy token receives the sum of their values.
    """

    name = "pytt_tok2vec"

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def from_pretrained(cls, vocab, name, **cfg):
        """Create a PyTT_TokenVectorEncoder instance using pre-trained weights
        from a PyTorch Transformer model, even if it's not installed as a
        spaCy package.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
        name (unicode): Name of pre-trained model, e.g. 'bert-base-uncased'.
        RETURNS (PyTT_TokenVectorEncoder): The token vector encoder.
        """
        cfg["from_pretrained"] = True
        cfg["pytt_name"] = name
        model = cls.Model(**cfg)
        self = cls(vocab, model=model, **cfg)
        return self

    @classmethod
    def Model(cls, **cfg):
        """Create an instance of `PyTT_Wrapper`, which holds the
        PyTorch-Transformers model.

        **cfg: Optional config parameters.
        RETURNS (thinc.neural.Model): The wrapped model.
        """
        name = cfg.get("pytt_name")
        if not name:
            raise ValueError("Need pytt_name argument, e.g. 'bert-base-uncased'")
        if cfg.get("from_pretrained"):
            model = PyTT_Wrapper.from_pretrained(name)
        else:
            model = PyTT_Wrapper(name)
        nO = model.nO
        batch_by_length = cfg.get("batch_by_length", 10)
        with Model.define_operators({">>": chain}):
            model = foreach_sentence(
                get_word_pieces
                >> with_length_batching(model >> get_last_hidden_state, batch_by_length)
            )
        model.nO = nO
        return model

    def __init__(self, vocab, model=True, **cfg):
        """Initialize the component.

        model (thinc.neural.Model / True): The component's model or `True` if
            not initialized yet.
        **cfg: Optional config parameters.
        """
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    def __call__(self, doc):
        """Process a Doc and assign the extracted features.

        doc (spacy.tokens.Doc): The Doc to process.
        RETURNS (spacy.tokens.Doc): The processed Doc.
        """
        self.require_model()
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream, batch_size=128):
        """Process Doc objects as a stream and assign the extracted features.

        stream (iterable): A stream of Doc objects.
        batch_size (int): The number of texts to buffer.
        YIELDS (spacy.tokens.Doc): Processed Docs in order.
        """
        for docs in minibatch(stream, size=batch_size):
            docs = list(docs)
            outputs = self.predict(docs)
            self.set_annotations(docs, outputs)
            for doc in docs:
                yield doc

    def begin_update(self, docs, drop=None, **cfg):
        """Get the predictions and a callback to complete the gradient update.
        This is only used internally within PyTT_Language.update.
        """
        outputs, backprop = self.model.begin_update(docs, drop=drop)

        def finish_update(docs, sgd=None):
            gradients = []
            for doc in docs:
                gradients.append(doc._.pytt_d_last_hidden_state)
                #gradients.append(
                #    Activations(
                #        doc._.pytt_d_last_hidden_state,
                #        doc._.pytt_d_pooler_output,
                #        doc._.pytt_d_all_hidden_states,
                #        doc._.pytt_d_all_attentions,
                #        is_grad=True,
                #    )
                #)
            backprop(gradients, sgd=sgd)
            for doc in docs:
                doc._.pytt_d_last_hidden_state.fill(0)
            return None

        return outputs, finish_update

    def predict(self, docs):
        """Run the transformer model on a batch of docs and return the
        extracted features.

        docs (iterable): A batch of Docs to process.
        RETURNS (list): A list of Activations objects, one per doc.
        """
        return self.model.predict(docs)

    def set_annotations(self, docs, activations):
        """Assign the extracted features to the Doc objects and overwrite the
        vector and similarity hooks.

        docs (iterable): A batch of `Doc` objects.
        activations (iterable): A batch of activations.
        """
        for doc, wp_tensor in zip(docs, activations):
            if doc.tensor is None:
                doc.tensor = self.model.ops.allocate((len(doc), self.model.nO))
            else:
                doc.tensor.fill(0)
            doc._.pytt_last_hidden_state = wp_tensor
            assert wp_tensor.shape == (
                len(doc._.pytt_word_pieces),
                self.model.nO,
            ), wp_tensor.shape
            # Count how often each word-piece token is represented. This allows
            # a weighted sum, so that we can make sure doc.tensor.sum()
            # equals wp_tensor.sum().
            align_sizes = [0 for _ in range(len(doc._.pytt_word_pieces))]
            for word_piece_slice in doc._.pytt_alignment:
                for i in word_piece_slice:
                    align_sizes[i] += 1
            for i, word_piece_slice in enumerate(doc._.pytt_alignment):
                for j in word_piece_slice:
                    doc.tensor[i] += wp_tensor[j] / align_sizes[j]
            # To make this weighting work, we "align" the boundary tokens against
            # every token in their sentence.
            for sent in doc.sents:
                cls_vector = wp_tensor[sent._.pytt_start]
                sep_vector = wp_tensor[sent._.pytt_end]
                doc.tensor[sent.start : sent.end + 1] += cls_vector / len(sent)
                doc.tensor[sent.start : sent.end + 1] += sep_vector / len(sent)
            doc.user_hooks["vector"] = get_doc_vector_via_tensor
            doc.user_span_hooks["vector"] = get_span_vector_via_tensor
            doc.user_token_hooks["vector"] = get_token_vector_via_tensor
            doc.user_hooks["similarity"] = get_similarity_via_tensor
            doc.user_span_hooks["similarity"] = get_similarity_via_tensor
            doc.user_token_hooks["similarity"] = get_similarity_via_tensor


def get_doc_vector_via_tensor(doc):
    return doc.tensor.sum(axis=0)


def get_span_vector_via_tensor(span):
    return span.doc.tensor[span.start : span.end].sum(axis=0)


def get_token_vector_via_tensor(token):
    return token.doc.tensor[token.i]


def get_similarity_via_tensor(doc1, doc2):
    v1 = doc1.vector
    v2 = doc2.vector
    xp = get_array_module(v1)
    return xp.dot(v1, v2) / (doc1.vector_norm * doc2.vector_norm)


@layerize
def get_word_pieces(sents, drop=0.0):
    assert isinstance(sents[0], Span)
    outputs = []
    for sent in sents:
        wp_start = sent._.pytt_start
        wp_end = sent._.pytt_end
        if wp_start is not None and wp_end is not None:
            outputs.append(sent.doc._.pytt_word_pieces[wp_start:wp_end+1])
        else:
            # Empty slice.
            outputs.append(sent.doc._.pytt_word_pieces[0:0])
    return outputs, None


@layerize
def get_last_hidden_state(activations, drop=0.0):
    def backprop_last_hidden_state(d_last_hidden_state, sgd=None):
        return d_last_hidden_state
    return activations.last_hidden_state, backprop_last_hidden_state


def without_length_batching(model, _):
    def apply_model_padded(inputs, drop=0.0):
        X = pad_batch(inputs)
        Y, get_dX = model.begin_update(X, drop=drop)
        outputs = [Y[i, :len(seq)] for i, seq in enumerate(inputs)]

        def backprop_batched(d_outputs, sgd=None):
            dY = pad_batch(d_outputs)
            dY = dY.reshape(len(d_outputs), -1, dY.shape[-1])
            dX = get_dX(dY, sgd=sgd)
            if dX is not None:
                d_inputs = [dX[i, :len(seq)] for i, seq in enumerate(d_outputs)]
            else:
                d_inputs = None
            return d_inputs

        return outputs, backprop_batched

    return wrap(apply_model_padded, model)


def with_length_batching(model, min_batch):
    """Wrapper that applies a model to variable-length sequences by first batching
    and padding the sequences. This allows us to group similarly-lengthed sequences
    together, making the padding less wasteful. If min_batch==1, no padding will
    be necessary.
    """
    if min_batch < 1:
        return without_length_batching(model, min_batch)

    def apply_model_to_batches(inputs, drop=0.0):
        backprops = []
        batches = batch_by_length(inputs, min_batch)
        # Initialize this, so we can place the outputs back in order.
        outputs = [None for _ in inputs]
        for indices in batches:
            X = pad_batch([inputs[i] for i in indices])
            Y, get_dX = model.begin_update(X, drop=drop)
            backprops.append(get_dX)
            for i, j in enumerate(indices):
                outputs[j] = Y[i, : len(inputs[j])]

        def backprop_batched(d_outputs, sgd=None):
            d_inputs = [None for _ in inputs]
            for indices, get_dX in zip(batches, backprops):
                dY = pad_batch([d_outputs[i] for i in indices])
                dY = dY.reshape(len(indices), -1, dY.shape[-1])
                dX = get_dX(dY, sgd=sgd)
                if dX is not None:
                    for i, j in enumerate(indices):
                        # As above, put things back in order, unpad.
                        d_inputs[j] = dX[i, : len(d_outputs[j])]
            return d_inputs

        return outputs, backprop_batched

    return wrap(apply_model_to_batches, model)


def foreach_sentence(layer, drop_factor=1.0):
    """Map a layer across sentences (assumes spaCy-esque .sents interface)"""
    ops = layer.ops

    def sentence_fwd(docs, drop=0.0):
        sents = flatten_list(doc.sents for doc in docs)
        sent_acts, bp_sent_acts = layer.begin_update(sents, drop=drop)
        # sent_acts is List[array], one per sent. Go to List[List[array]],
        # then List[array] (one per doc).
        nested = unflatten_list(sent_acts, [len(list(doc.sents)) for doc in docs])
        # Need this for the callback.
        doc_sent_lengths = [[len(sa) for sa in doc_sa] for doc_sa in nested]
        doc_acts = [ops.flatten(doc_sa) for doc_sa in nested]
        assert len(docs) == len(doc_acts)

        def sentence_bwd(d_doc_acts, sgd=None):
            d_nested = [
                ops.unflatten(d_doc_acts[i], doc_sent_lengths[i])
                for i in range(len(d_doc_acts))
            ]
            d_sent_acts = flatten_list(d_nested)
            d_sents = bp_sent_acts(d_sent_acts, sgd=sgd)
            if d_sents is None or all(ds is None for ds in d_sents):
                return None
            # Finally we have List[array], one per sentence, where each row is
            # the gradient wrt the token.
            # We want List[array], one per document.
            # First get List[List[array]], grouped by doc, then just flatten
            # the lists.
            n_sents = [len(L) for L in doc_sent_lengths]
            return [ops.flatten(ds) for ds in unflatten_list(d_sents, n_sents)]

        return doc_acts, sentence_bwd

    model = wrap(sentence_fwd, layer)
    return model
