from typing import Any, List
from thinc.neural.ops import get_array_module
from spacy.pipeline import Tok2Vec
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import minibatch

from ..wrapper import TransformersWrapper
from ..model_registry import get_model_function
from ..activations import Activations, RaggedArray
from ..util import get_config, get_model, get_sents, PIPES, ATTRS


# TODO: Add assignments for extension attributes.
@component("trf_tok2vec", assigns=["doc.tensor"])
class TransformersTok2Vec(Tok2Vec):
    """spaCy pipeline component to use transformer models.

    The component assigns the output of the transformer to the Doc's
    extension attributes. We also calculate an alignment between the word-piece
    tokens and the spaCy tokenization, so that we can use the last hidden states
    to set the doc.tensor attribute. When multiple word-piece tokens align to
    the same spaCy token, the spaCy token receives the sum of their values.
    """

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
