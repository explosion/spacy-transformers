from .language import TransformersLanguage
from .pipeline.tok2vec import TransformersTok2Vec  # noqa
from .pipeline.textcat import TransformersTextCategorizer  # noqa
from .pipeline.wordpiecer import TransformersWordPiecer  # noqa
from .model_registry import register_model, get_model_function  # noqa
from .about import __version__  # noqa

TransformersLanguage.install_extensions()
