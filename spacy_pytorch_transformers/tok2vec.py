from spacy.pipeline import Pipe
from thinc.api import chain, layerize, wrap
from thinc.neural.ops import get_array_module
from spacy.util import minibatch
from spacy.tokens import Span

from .wrapper import PyTT_Wrapper
from .util import batch_by_length, pad_batch, flatten_list, unflatten_list, get_sents


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
        batch_by_length = cfg.get("batch_by_length", 1)
        model = with_length_batching(model, batch_by_length)
        model = chain(get_word_pieces, model)
        model = foreach_sentence(model)
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
            yield from docs

    def begin_update(self, docs, drop=None, **cfg):
        """Get the predictions and a callback to complete the gradient update.
        This is only used internally within PyTT_Language.update.
        """
        outputs, backprop = self.model.begin_update(docs, drop=drop)

        def finish_update(docs, sgd=None):
            gradients = [doc._.pytt_gradients.last_hidden_state for doc in docs]
            backprop(gradients, sgd=sgd)
            for doc in docs:
                doc._.pytt_outputs = None
            return None

        return outputs, finish_update

    def predict(self, docs):
        """Run the transformer model on a batch of docs and return the
        extracted features.

        docs (iterable): A batch of Docs to process.
        RETURNS (namedtuple): Named tuple containing the outputs.
        """
        outputs = self.model.predict(docs)
        for sent_outs in outputs:
            for out in sent_outs:
                assert out.has_last_hidden_state is not None
        return outputs

    def set_annotations(self, docs, activations):
        """Assign the extracted features to the Doc objects and overwrite the
        vector and similarity hooks.

        docs (iterable): A batch of `Doc` objects.
        activations (iterable): A batch of activations.
        """
        for doc, sent_acts in zip(docs, activations):
            doc.tensor = self.model.ops.allocate((len(doc), self.model.nO))
            sents = get_sents(doc)
            assert len(sents) == len(sent_acts)
            for i, (sent, sent_acts) in enumerate(zip(sents, sent_acts)):
                sent._.pytt_outputs = sent_acts
                wp_tensor = sent_acts.last_hidden_state
                print(wp_tensor.shape)
                # Count how often each word-piece token is represented. This allows
                # a weighted sum, so that we can make sure doc.tensor.sum()
                # equals wp_tensor[1:-1].sum().
                align_sizes = [0 for _ in range(len(sent._.pytt_word_pieces))]
                for word_piece_slice in sent._.pytt_alignment:
                    for i in word_piece_slice:
                        align_sizes[i] += 1
                for i, word_piece_slice in enumerate(sent._.pytt_alignment):
                    for j in word_piece_slice:
                        doc.tensor[sent.start + i] += wp_tensor[j] / max(
                            1, align_sizes[j]
                        )
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
    return [sent._.pytt_word_pieces for sent in sents], None


def with_length_batching(model, min_batch):
    """Wrapper that applies a model to variable-length sequences by first batching
    and padding the sequences. This allows us to group similarly-lengthed sequences
    together, making the padding less wasteful. If min_batch==1, no padding will
    be necessary.
    """

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
                outputs[j] = Y.get_slice(i, slice(0, len(inputs[j])))

        def backprop_batched(d_outputs, sgd=None):
            d_inputs = [None for _ in inputs]
            for indices, get_dX in zip(batches, backprops):
                dY = pad_batch([d_outputs[i] for i in indices])
                dX = get_dX(dY, sgd=sgd)
                if dX is not None:
                    for i, j in enumerate(indices):
                        # As above, put things back in order, unpad.
                        # Note that there's no fields to deal with here, as
                        # the input doesn't have any.
                        d_inputs[j] = dX[i, : len(d_outputs[j])]
            return d_inputs

        return outputs, backprop_batched

    return wrap(apply_model_to_batches, model)


def foreach_sentence(layer, drop_factor=1.0):
    """Map a layer across sentences (assumes spaCy-esque .sents interface)"""

    def sentence_fwd(docs, drop=0.0):
        sents = []
        lengths = []
        for doc in docs:
            doc_sents = get_sents(doc)
            sents.extend(doc_sents)
            lengths.append(len(doc_sents))
        flat, bp_flat = layer.begin_update(sents, drop=drop)
        outputs = unflatten_list(flat, lengths)
        assert len(docs) == len(outputs)

        def sentence_bwd(d_output, sgd=None):
            d_flat = bp_flat(flatten_list(d_output), sgd=sgd)
            if d_flat is None:
                return d_flat
            return unflatten_list(d_flat, lengths)

        return outputs, sentence_bwd

    model = wrap(sentence_fwd, layer)
    return model
