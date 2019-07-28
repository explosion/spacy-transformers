from .language import PyTT_Language
from .tok2vec import PyTT_TokenVectorEncoder  # noqa
from .pipeline import PyTT_TextCategorizer  # noqa
from .wordpiecer import PyTT_WordPiecer  # noqa
from .about import __version__  # noqa

PyTT_Language.install_extensions()
