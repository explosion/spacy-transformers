from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class

from . import about
from .util import align_word_pieces


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
        for cls in [Doc, Span, Token]:
            cls.set_extension("pytt_alignment", default=None)
            cls.set_extension("pytt_word_pieces", default=None)
            cls.set_extension("pytt_word_pieces_", default=None)
            cls.set_extension("pytt_outputs", default=None)
            cls.set_extension("pytt_gradients", default=None)

    def __init__(
        self, vocab=True, make_doc=True, max_length=10 ** 6, meta={}, **kwargs
    ):
        meta["lang_factory"] = self.lang_factory_name
        # Add this package to requirements to it will be included in the
        # install_requires of any model using this language class
        package = f"{about.__title__}>={about.__version__}"
        meta.setdefault("requirements", []).append(package)
        self.lang = meta.get("lang", "xx")
        self.Defaults = get_defaults(self.lang)
        super().__init__(vocab, make_doc, max_length, meta=meta, **kwargs)
        if meta.get("name"):
            pytt_cls = get_pytt_tokenizer(meta["name"])
            self.pytt_tokenizer = pytt_cls.from_pretrained(pytt_name)
        else:
            self.pytt_tokenizer = None

    def make_doc(self, text):
        doc = self.tokenizer(text)
        bos = self.pytt_tokenizer.cls_token
        sep = self.pytt_tokenizer.sep_token
        strings = [bos] + self.pytt_tokenizer.tokenize(text) + [sep]
        doc._.pytt_word_pieces_ = strings
        doc._.pytt_word_pieces = self.pytt_tokenizer.convert_tokens_to_ids(strings)
        doc._.pytt_alignment = align_word_pieces([w.text for w in doc], strings)
        return doc

    def update(self, docs, golds, drop=0.0, sgd=None, losses=None, component_cfg=None):
        tok2vec = self.get_pipe("pytt_tok2vec")
        docs, backprop_tok2vec = tok2vec.begin_update(
            docs, drop=drop, **component_cfg.get("pytt_tok2vec", {})
        )
        with self.disable_pipes("pytt_tok2vec"):
            super().update(
                self,
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
