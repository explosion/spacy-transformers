from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class

from . import about
from .util import get_sents


class PyTT_Language(Language):
    """A subclass of spacy.Language that holds a PyTorch-Transformer (PyTT) pipeline.

    PyTT pipelines work only slightly differently from spaCy's default pipelines.
    Specifically, we introduce a new pipeline component at the start of the pipeline,
    PyTT_TokenVectorEncoder. We then modify the nlp.update() function to run
    the PyTT_TokenVectorEncoder before the other pipeline components, and
    backprop it after the other components are done.

    The PyTT_Language class expects the following extension attributes:

    * doc._.pytt_word_pieces: A Torch tensor of word-piece IDs.
    * doc._.pytt_word_pieces_: The string forms of the word-piece IDs.
    * doc._.pytt_outputs: All outputs produced by the PyTorch Transformer model.
    * doc._.pytt_gradients: Gradients of the pytt_outputs. These get incremented
        during nlp.update(), and then cleared at the end once the update is made.
    """

    lang_factory_name = "pytt"

    @staticmethod
    def install_extensions():
        attrs = ["pytt_alignment", "pytt_word_pieces", "pytt_word_pieces_"]
        for attr in attrs:
            Span.set_extension(attr, default=None)
            Doc.set_extension(attr, getter=get_doc_getter(attr))
            Token.set_extension(attr, getter=get_token_getter(attr))
        for attr in ["pytt_outputs", "pytt_gradients"]:
            Span.set_extension(attr, default=None)
            Doc.set_extension(attr, getter=get_doc_getter(attr))

    def __init__(
        self, vocab=True, make_doc=True, max_length=10 ** 6, meta={}, **kwargs
    ):
        """Initialize the language class. Expects either a pytt_name setting in
        the meta or as a keyword argument, specifying the pre-trained model
        name. This is used to set up the model-specific tokenizer.
        """
        meta = dict(meta)
        meta["lang_factory"] = self.lang_factory_name
        # Add this package to requirements to it will be included in the
        # install_requires of any model using this language class
        package = f"{about.__title__}>={about.__version__}"
        meta.setdefault("requirements", []).append(package)
        self.lang = meta.get("lang", "xx")
        self.Defaults = get_defaults(self.lang)
        super().__init__(vocab, make_doc, max_length, meta=meta, **kwargs)

    def update(self, docs, golds, drop=0.0, sgd=None, losses=None, component_cfg={}):
        component_cfg = dict(component_cfg)
        components = ["pytt_wordpiecer", "sentencizer"]
        with self.disable_pipes(*[p for p in self.pipe_names if p not in components]):
            docs = [self(doc) if isinstance(doc, str) else doc for doc in docs]
        tok2vec = self.get_pipe("pytt_tok2vec")
        pytt_outputs, backprop_tok2vec = tok2vec.begin_update(
            docs, drop=drop, **component_cfg.get("pytt_tok2vec", {})
        )
        tok2vec.set_annotations(docs, pytt_outputs)
        for doc in docs:
            assert doc._.pytt_outputs
        components = [p for p in components if p in self.pipe_names]
        with self.disable_pipes("pytt_tok2vec", *components):
            super().update(
                docs,
                golds,
                drop=0.0,
                sgd=sgd,
                losses=losses,
                component_cfg=component_cfg,
            )
        backprop_tok2vec(docs, sgd=sgd)


def get_defaults(lang):
    """Get the language-specific defaults, if available in spaCy."""
    try:
        lang_cls = get_lang_class(lang)
        return lang_cls.Defaults
    except ImportError:
        return Language.Defaults


def get_doc_getter(attr):
    def doc_getter(doc):
        values = [sent._.get(attr) for sent in get_sents(doc)]
        if all(v is None for v in values):
            return None
        else:
            return values
    return doc_getter


def get_token_getter(attr):
    def token_getter(token):
        span = token.sent if token.doc.is_sentenced else token.doc[0:-1]
        values = span._.get(attr)
        if values is None:
            return None
        else:
            return values[token.i - span.start]
    return token_getter
