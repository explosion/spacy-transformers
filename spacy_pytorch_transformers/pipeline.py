from collections import namedtuple
import spacy.pipeline
from spacy._ml import build_simple_cnn_text_classifier
from thinc.api import layerize


class PyTT_TextCategorizer(spacy.pipeline.TextCategorizer):
    """Subclass of spaCy's built-in TextCategorizer component that supports
    using the features assigned by the PyTorch-Transformers models via the token
    vector encoder. It requires the PyTT_TokenVectorEncoder to run before it in
    the pipeline.
    """

    name = "pytt_textcat"

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        return cls(nlp.vocab, **cfg)

    @classmethod
    def Model(cls, nr_class, exclusive_classes=False, **cfg):
        """Create a text classification model using a PyTorch-Transformers model
        for token vector encoding.

        nr_class (int): Number of classes.
        width (int): The width of the tensors being assigned.
        exclusive_classes (bool): Make categories mutually exclusive.
        **cfg: Optional config parameters.
        RETURNS (thinc.neural.Model): The model.
        """
        tok2vec = layerize(get_pytt_last_hidden)
        tok2vec.nO = cfg["token_vector_width"]
        return build_simple_cnn_text_classifier(
            tok2vec, nr_class=nr_class, width=tok2vec.nO, **cfg
        )


def get_pytt_last_hidden(docs, drop=0.0):
    """Function that can be wrapped as a pipeline component, that gets the
    pytt_last_hidden extension attribute from a batch of Doc objects. During
    the backward pass, we accumulate the gradients into doc._.pytt_gradients.
    """
    outputs = [doc._.pytt_outputs.last_hidden_state for doc in docs]

    def backprop_pytt_last_hidden(d_outputs, sgd=None):
        for doc, gradient in zip(docs, d_outputs):
            if doc._.pytt_gradients is None:
                col_names = doc._.pytt_outputs._fields
                Grads = namedtuple("pytt_gradients", col_names)
                doc._.pytt_gradients = Grads(last_hidden_state=gradient)
            else:
                doc._.pytt_gradients.last_hidden_state += gradient
        return None

    return outputs, backprop_pytt_last_hidden
