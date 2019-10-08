from .language import TransformersLanguage
from .pipeline.tok2vec import TransformersTok2Vec  # noqa
from .pipeline.textcat import TransformersTextCategorizer  # noqa
from .pipeline.wordpiecer import TransformersWordPiecer  # noqa
from .model_registry import register_model, get_model_function  # noqa
from .util import pkg_meta

__version__ = pkg_meta["version"]
TransformersLanguage.install_extensions()
