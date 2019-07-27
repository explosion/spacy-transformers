from collections import namedtuple
from spacy.pipeline import Pipe
from thinc.api import chain, foreach_sentence, layerize, wrap
from thinc.neural.ops import get_array_module
from spacy.util import minibatch

from .wrapper import PyTT_Wrapper
from .util import batch_by_length, pad_batch


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
        if cfg.get("batch_by_length"):
            model = with_length_batching(model, cfg["batch_by_length"])
        model = chain(get_word_pieces, model)
        if cfg.get("per_sentence"):
            model = foreach_sentence(model)
        model.nO = nO
        return model

    def __init__(self, vocab, model=True, **cfg):
        """Initialize the component.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
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
        self.set_annotations(docs, outputs)

        def finish_update(docs, sgd=None):
            gradients = [doc._.pytt_gradients for doc in docs]
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
        for out in outputs:
            assert out.last_hidden_state is not None
        return outputs

    def set_annotations(self, docs, outputs):
        """Assign the extracted features to the Doc objects and overwrite the
        vector and similarity hooks.

        docs (iterable): A batch of `Doc` objects.
        outputs (iterable): A batch of outputs.
        """
        for doc, output in zip(docs, outputs):
            doc._.pytt_outputs = output
            doc.tensor = self.model.ops.allocate((len(doc), self.model.nO))
            wp_tensor = output.last_hidden_state
            # Count how often each word-piece token is represented. This allows
            # a weighted sum, so that we can make sure doc.tensor.sum()
            # equals wp_tensor[1:-1].sum().
            align_sizes = [0 for _ in range(len(doc._.pytt_word_pieces))]
            for word_piece_slice in doc._.pytt_alignment:
                for i in word_piece_slice:
                    align_sizes[i] += 1
            for i, word_piece_slice in enumerate(doc._.pytt_alignment):
                for j in word_piece_slice:
                    doc.tensor[i] += wp_tensor[j] / max(1, align_sizes[j])
            doc.user_hooks["vector"] = get_doc_vector_via_tensor
            doc.user_span_hooks["vector"] = get_span_vector_via_tensor
            doc.user_token_hooks["vector"] = get_token_vector_via_tensor
            doc.user_hooks["similarity"] = get_similarity_via_tensor
            doc.user_span_hooks["similarity"] = get_similarity_via_tensor
            doc.user_token_hooks["similarity"] = get_similarity_via_tensor

def get_doc_vector_via_tensor(doc):
    return doc.tensor.sum(axis=0)

def get_span_vector_via_tensor(span):
    return span.doc.tensor[span.start:span.end].sum(axis=0)

def get_token_vector_via_tensor(token):
    return token.doc.tensor[token.i]

def get_similarity_via_tensor(doc1, doc2):
    v1 = doc1.vector
    v2 = doc2.vector
    xp = get_array_module(v1)
    return xp.dot(v1, v2) / (doc1.vector_norm * doc2.vector_norm)


@layerize
def get_word_pieces(docs, drop=0.0):
    return [doc._.pytt_word_pieces for doc in docs], None


def with_length_batching(model, min_batch):
    """Wrapper that applies a model to variable-length sequences by first batching
    and padding the sequences. This allows us to group similarly-lengthed sequences
    together, making the padding less wasteful. If min_batch==1, no padding will
    be necessary.
    """
    col_names = getattr(model, "out_cols", [None])

    def apply_model_to_batches(inputs, drop=0.0):
        backprops = []
        batches = batch_by_length(inputs, min_batch)
        # Initialize this, so we can place the outputs back in order.
        outputs = [[None for _ in col_names] for _ in inputs]
        for indices in batches:
            X = pad_batch([inputs[i] for i in indices])
            Y, get_dX = model.begin_update(X, drop=drop)
            backprops.append(get_dX)
            for col in range(len(col_names)):
                for i, j in enumerate(indices):
                    if Y[col][i] is not None:
                        # The index j tells us where the row was.
                        # We also need to remember to unpad.
                        outputs[j][col] = Y[col][i, : len(inputs[j])]

        def backprop_batched(d_outputs, sgd=None):
            d_inputs = [None for _ in inputs]
            for indices, get_dX in zip(batches, backprops):
                dY = pad_batch([d_outputs[i] for i in indices])
                dX = get_dX(dY, sgd=sgd)
                if dX is not None:
                    for i, j in enumerate(indices):
                        # As above, put things back in order, unpad.
                        # Note that there's no columns to deal with here, as
                        # the input doesn't have any.
                        d_inputs[j] = dX[i, : len(d_outputs[j])]
            return d_inputs

        if col_names == [None]:
            outputs = [o[0] for o in outputs]
        else:
            MakeOutput = namedtuple("pytt_outputs", col_names)
            outputs = [MakeOutput(*o) for o in outputs]
        return outputs, backprop_batched

    return wrap(apply_model_to_batches, model)
