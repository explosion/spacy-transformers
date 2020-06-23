from typing import Any, List
from thinc.neural.ops import get_array_module
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import minibatch

from ..wrapper import TransformersWrapper
from ..model_registry import get_model_function
from ..activations import Activations, RaggedArray
from ..util import get_config, get_model, get_sents, PIPES, ATTRS
from ..util import get_config_name


class TransformersTok2Vec(Pipe):
    """spaCy pipeline component to use transformer models.

    The component assigns the output of the transformer to the Doc's
    extension attributes. We also calculate an alignment between the word-piece
    tokens and the spaCy tokenization, so that we can use the last hidden states
    to set the doc.tensor attribute. When multiple word-piece tokens align to
    the same spaCy token, the spaCy token receives the sum of their values.
    """

    name = PIPES.tok2vec
    factory = PIPES.tok2vec

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def from_pretrained(cls, vocab: Vocab, name: str, **cfg):
        """Create a TransformersTok2Vec instance using pre-trained weights
        from a transformer model, even if it's not installed as a
        spaCy package.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
        name (unicode): Name of pre-trained model, e.g. 'bert-base-uncased'.
        RETURNS (TransformersTok2Vec): The token vector encoder.
        """
        cfg["trf_name"] = name
        model = cls.Model(from_pretrained=True, **cfg)
        cfg["trf_config"] = dict(model._model.transformers_model.config.to_dict())
        self = cls(vocab, model=model, **cfg)
        return self

    @classmethod
    def Model(cls, **cfg) -> Any:
        """Create an instance of `TransformersWrapper`, which holds the
        Transformers model.

        **cfg: Optional config parameters.
        RETURNS (thinc.neural.Model): The wrapped model.
        """
        name = cfg.get("trf_name")
        if not name:
            raise ValueError(f"Need trf_name argument, e.g. 'bert-base-uncased'")
        cfg["trf_model_type"] = get_config_name(cfg.get("trf_name"))
        if cfg.get("from_pretrained"):
            wrap = TransformersWrapper.from_pretrained(name)
        else:
            config = cfg["trf_config"]
            # Work around floating point limitation in ujson:
            # If we have the setting "layer_norm_eps" as 0,
            # that's because of misprecision in serializing. Fix that.
            config["layer_norm_eps"] = 1e-12
            config_cls = get_config(name)
            model_cls = get_model(name)
            # Need to match the name their constructor expects.
            if "vocab_size" in cfg["trf_config"]:
                vocab_size = cfg["trf_config"]["vocab_size"]
                cfg["trf_config"]["vocab_size_or_config_json_file"] = vocab_size
            wrap = TransformersWrapper(name, config, model_cls(config_cls(**config)))
        make_model = get_model_function(cfg.get("architecture", "tok2vec_per_sentence"))
        model = make_model(wrap, cfg)
        setattr(model, "nO", wrap.nO)
        setattr(model, "_model", wrap)
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
    def transformers_model(self):
        return self.model._model.transformers_model

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
        This is only used internally within TransformersLanguage.update.
        """
        outputs, backprop = self.model.begin_update(docs, drop=drop)

        def finish_update(docs, sgd=None):
            assert len(docs)
            d_lh = []
            d_po = []
            lh_lengths = []
            po_lengths = []
            for doc in docs:
                d_lh.append(doc._.get(ATTRS.d_last_hidden_state))
                d_po.append(doc._.get(ATTRS.d_pooler_output))
                lh_lengths.append(doc._.get(ATTRS.d_last_hidden_state).shape[0])
                po_lengths.append(doc._.get(ATTRS.d_pooler_output).shape[0])
            xp = self.model.ops.xp
            gradients = Activations(
                RaggedArray(xp.vstack(d_lh), lh_lengths),
                RaggedArray(xp.vstack(d_po), po_lengths),
            )
            backprop(gradients, sgd=sgd)
            for doc in docs:
                doc._.get(ATTRS.d_last_hidden_state).fill(0)
                doc._.get(ATTRS.d_last_hidden_state).fill(0)
            return None

        return outputs, finish_update

    def predict(self, docs):
        """Run the transformer model on a batch of docs and return the
        extracted features.

        docs (iterable): A batch of Docs to process.
        RETURNS (list): A list of Activations objects, one per doc.
        """
        return self.model.predict(docs)

    def set_annotations(self, docs: List[Doc], activations: Activations):
        """Assign the extracted features to the Doc objects and overwrite the
        vector and similarity hooks.

        docs (iterable): A batch of `Doc` objects.
        activations (iterable): A batch of activations.
        """
        xp = activations.xp
        for i, doc in enumerate(docs):
            # Make it 2d -- acts are always 3d, to represent batch size.
            wp_tensor = activations.lh.get(i)
            doc.tensor = self.model.ops.allocate((len(doc), self.model.nO))
            doc._.set(ATTRS.last_hidden_state, wp_tensor)
            if activations.has_po:
                pooler_output = activations.po.get(i)
                doc._.set(ATTRS.pooler_output, pooler_output)
            doc._.set(
                ATTRS.d_last_hidden_state, xp.zeros((0, 0), dtype=wp_tensor.dtype)
            )
            doc._.set(ATTRS.d_pooler_output, xp.zeros((0, 0), dtype=wp_tensor.dtype))
            doc._.set(ATTRS.d_all_hidden_states, [])
            doc._.set(ATTRS.d_all_attentions, [])
            if wp_tensor.shape != (len(doc._.get(ATTRS.word_pieces)), self.model.nO):
                raise ValueError(
                    "Mismatch between tensor shape and word pieces. This usually "
                    "means we did something wrong in the sentence reshaping, "
                    "or possibly finding the separator tokens."
                )
            # Count how often each word-piece token is represented. This allows
            # a weighted sum, so that we can make sure doc.tensor.sum()
            # equals wp_tensor.sum(). Do this with sensitivity to boundary tokens
            wp_rows, align_sizes = _get_boundary_sensitive_alignment(doc)
            wp_weighted = wp_tensor / xp.array(align_sizes, dtype="f").reshape((-1, 1))
            # TODO: Obviously incrementing the rows individually is bad. How
            # to do in one shot without blowing up the memory?
            for i, word_piece_slice in enumerate(wp_rows):
                doc.tensor[i] = wp_weighted[word_piece_slice,].sum(0)
            doc.user_hooks["vector"] = get_doc_vector_via_tensor
            doc.user_span_hooks["vector"] = get_span_vector_via_tensor
            doc.user_token_hooks["vector"] = get_token_vector_via_tensor
            doc.user_hooks["similarity"] = get_similarity_via_tensor
            doc.user_span_hooks["similarity"] = get_similarity_via_tensor
            doc.user_token_hooks["similarity"] = get_similarity_via_tensor


def _get_boundary_sensitive_alignment(doc):
    align_sizes = [0 for _ in range(len(doc._.get(ATTRS.word_pieces)))]
    wp_rows = []
    for word_piece_slice in doc._.get(ATTRS.alignment):
        wp_rows.append(list(word_piece_slice))
        for i in word_piece_slice:
            align_sizes[i] += 1
    # To make this weighting work, we "align" the boundary tokens against
    # every token in their sentence. The boundary tokens are otherwise
    # unaligned, which is how we identify them.
    for sent in get_sents(doc):
        offset = sent._.get(ATTRS.start)
        for i in range(len(sent._.get(ATTRS.word_pieces))):
            if align_sizes[offset + i] == 0:
                align_sizes[offset + i] = len(sent)
                for tok in sent:
                    wp_rows[tok.i].append(offset + i)
    return wp_rows, align_sizes


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
