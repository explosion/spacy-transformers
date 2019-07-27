from collections import namedtuple
import spacy.pipeline
from spacy._ml import build_simple_cnn_text_classifier
from thinc.api import layerize


class PyTT_TextCategorizer(spacy.pipeline.TextCategorizer):
    name = "pytt_textcat"

    @classmethod
    def Model(cls, nr_class, width, exclusive_classes=False, **cfg):
        tok2vec = layerize(get_pytt_last_hidden)
        tok2vec.nO = cfg["token_vector_width"]
        return build_simple_cnn_text_classifier(
            tok2vec, nr_class=nr_class, width=width, **cfg
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
