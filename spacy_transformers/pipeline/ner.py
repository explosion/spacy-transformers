import spacy.pipeline
from ..model_registry import get_model_function
from ..util import PIPES

DEBUG_LOSS = False


class TransformersEntityRecognizer(spacy.pipeline.EntityRecognizer):
    """Subclass of spaCy's built-in EntityRecognizer component that supports
    using the features assigned by the PyTorch-Transformers models via the token
    vector encoder. It requires the TransformerTokenVectorEncoder to run before
    it in the pipeline.
    """

    name = PIPES.ner
    factory = PIPES.ner

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def Model(cls, nr_class, **cfg):
        hidden_width = cfg.get("hidden_width", 64)
        token_vector_width = cfg.get("token_vector_width", 96)
        nr_feature = cls.nr_feature
        maxout_pieces = cfg.get("maxout_pieces", 3)
        tensor_size = cfg.get("tensor_size", 768)

        configs = {
            "tok2vec": {
                "arch": "tensor_affine_tok2vec",
                "config": {
                    "output_size": token_vector_width,
                    "tensor_size": tensor_size
                }
            },
            "lower": {
                "arch": "precomputable_maxout",
                "config": {
                    "output_size": hidden_width,
                    "input_size": token_vector_width,
                    "number_features": nr_feature,
                    "maxout_pieces": maxout_pieces
                }
            },
            "upper": {
                "arch": "affine_output",
                "config": {
                    "output_size": nr_class,
                    "input_size": hidden_width,
                    "drop_factor": 0.0
                }
            }`
        }
        tok2vec_arch = get_model_function(configs["tok2vec"]["arch"])
        lower_arch = get_model_function(configs["lower"]["arch"])
        upper_arch = get_model_function(configs["upper"]["arch"])
        tok2vec = tok2vec_arch(**configs["tok2vec"]["config"])
        lower = lower_arch(**configs["lower"]["config"])

        with Model.use_device('cpu'):
            upper = upper_arch(**configs["upper"]["config"])
        upper.W *= 0
        return ParserModel(tok2vec, lower, upper), configs
