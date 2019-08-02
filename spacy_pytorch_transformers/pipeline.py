import spacy.pipeline
from .model_registry import get_model_function


class PyTT_TextCategorizer(spacy.pipeline.TextCategorizer):
    """Subclass of spaCy's built-in TextCategorizer component that supports
    using the features assigned by the PyTorch-Transformers models via the token
    vector encoder. It requires the PyTT_TokenVectorEncoder to run before it in
    the pipeline.
    """

    name = "pytt_textcat"

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def Model(cls, nr_class=1, exclusive_classes=False, **cfg):
        """Create a text classification model using a PyTorch-Transformers model
        for token vector encoding.

        nr_class (int): Number of classes.
        width (int): The width of the tensors being assigned.
        exclusive_classes (bool): Make categories mutually exclusive.
        **cfg: Optional config parameters.
        RETURNS (thinc.neural.Model): The model.
        """
        make_model = get_model_function(
            cfg.get("architecture", "fine_tune_class_vector")
        )
        return make_model(nr_class, exclusive_classes=exclusive_classes, **cfg)
