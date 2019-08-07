from typing import List, Tuple, Callable, Any, Optional
from spacy.pipeline import Pipe
from thinc.neural.ops import get_array_module
from thinc.neural._classes.model import Model
from thinc.api import chain, layerize, wrap
from spacy.util import minibatch
from spacy.tokens import Doc, Span

from .wrapper import PyTT_Wrapper
from .util import get_pytt_config, get_pytt_model
from .util import batch_by_length, pad_batch, flatten_list, unflatten_list, Activations
from .util import pad_batch_activations
from .util import Array, Optimizer, Dropout


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
        cfg["pytt_name"] = name
        model = cls.Model(from_pretrained=True, **cfg)
        cfg["pytt_config"] = dict(model._model.pytt_model.config.to_dict())
        self = cls(vocab, model=model, **cfg)
        return self

    @classmethod
    def Model(cls, **cfg) -> Any:
        """Create an instance of `PyTT_Wrapper`, which holds the
        PyTorch-Transformers model.

        **cfg: Optional config parameters.
        RETURNS (thinc.neural.Model): The wrapped model.
        """
        name = cfg.get("pytt_name")
        if not name:
            raise ValueError("Need pytt_name argument, e.g. 'bert-base-uncased'")
        if cfg.get("from_pretrained"):
            pytt_model = PyTT_Wrapper.from_pretrained(name)
        else:
            # Work around floating point limitation in ujson:
            # If we have the setting cfg["pytt_config"]["layer_norm_eps"] as 0,
            # that's because of misprecision in serializing. Fix that.
            cfg["pytt_config"]["layer_norm_eps"] = 1e-12
            config_cls = get_pytt_config(name)
            model_cls = get_pytt_model(name)
            model = model_cls(config_cls(**cfg["pytt_config"]))
            pytt_model = PyTT_Wrapper(name, cfg["pytt_config"], model)
        nO = pytt_model.nO
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
        setattr(model, "nO", nO)
        setattr(model, "_model", pytt_model)
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

    @property
    def token_vector_width(self):
        return self.model._model.nO

    @property
    def pytt_model(self):
        return self.model._model.pytt_model

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
                gradients.append(
                    Activations(
                        doc._.pytt_d_last_hidden_state,
                        doc._.pytt_d_pooler_output,
                        [],
                        [],
                        is_grad=True,
                    )
                )
            backprop(gradients, sgd=sgd)
            for doc in docs:
                doc._.pytt_d_last_hidden_state.fill(0)
                doc._.pytt_d_pooler_output.fill(0)
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
        for doc, doc_acts in zip(docs, activations):
            xp = get_array_module(doc_acts.lh)
            wp_tensor = doc_acts.lh
            doc.tensor = self.model.ops.allocate((len(doc), self.model.nO))
            doc._.pytt_last_hidden_state = wp_tensor
            doc._.pytt_pooler_output = doc_acts.po
            doc._.pytt_all_hidden_states = doc_acts.ah
            doc._.pytt_all_attentions = doc_acts.aa
            doc._.pytt_d_last_hidden_state = xp.zeros((0,), dtype=wp_tensor.dtype)
            doc._.pytt_d_pooler_output = xp.zeros((0,), dtype=wp_tensor.dtype)
            doc._.pytt_d_all_hidden_states = []
            doc._.pytt_d_all_attentions = []
            if wp_tensor.shape != (len(doc._.pytt_word_pieces), self.model.nO):
                print("# word pieces: ", len(doc._.pytt_word_pieces))
                print("# tensor rows: ", wp_tensor.shape[0])
                for sent in doc.sents:
                    if sent._.pytt_start is None or sent._.pytt_end is None:
                        print("Text: ", sent.text)
                        print("WPs: ", sent._.pytt_word_pieces_)
                        print(sent._.pytt_start, sent._.pytt_end)
                raise ValueError(
                    "Mismatch between tensor shape and word pieces. This usually "
                    "means we did something wrong in the sentence reshaping, "
                    "or possibly finding the separator tokens."
                )
            # Count how often each word-piece token is represented. This allows
            # a weighted sum, so that we can make sure doc.tensor.sum()
            # equals wp_tensor.sum().
            # TODO: Obviously incrementing the rows individually is bad. Need
            # to make this more efficient. Maybe just copy to CPU, do our stuff,
            # copy back to GPU?
            align_sizes = [0 for _ in range(len(doc._.pytt_word_pieces))]
            for word_piece_slice in doc._.pytt_alignment:
                for i in word_piece_slice:
                    align_sizes[i] += 1
            for i, word_piece_slice in enumerate(doc._.pytt_alignment):
                for j in word_piece_slice:
                    doc.tensor[i] += wp_tensor[j] / align_sizes[j]
            # To make this weighting work, we "align" the boundary tokens against
            # every token in their sentence.
            if doc.tensor.sum() != wp_tensor.sum():
                for sent in doc.sents:
                    if sent._.pytt_start is not None and sent._.pytt_end is not None:
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
            outputs.append(sent.doc._.pytt_word_pieces[wp_start : wp_end + 1])
        else:
            # Empty slice.
            outputs.append(sent.doc._.pytt_word_pieces[0:0])
    return outputs, None



