from .language import PyTT_Language
from .pipeline.tok2vec import PyTT_TokenVectorEncoder  # noqa
from .pipeline.textcat import PyTT_TextCategorizer  # noqa
from .pipeline.wordpiecer import PyTT_WordPiecer  # noqa
from .model_registry import register_model, get_model_function  # noqa
from .about import __version__  # noqa

PyTT_Language.install_extensions()
