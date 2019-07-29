from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class

from .util import is_special_token
from . import about


class PyTT_Language(Language):
    """A subclass of spacy.Language that holds a PyTorch-Transformer (PyTT) pipeline.

    PyTT pipelines work only slightly differently from spaCy's default pipelines.
    Specifically, we introduce a new pipeline component at the start of the pipeline,
    PyTT_TokenVectorEncoder. We then modify the nlp.update() function to run
    the PyTT_TokenVectorEncoder before the other pipeline components, and
    backprop it after the other components are done.
    """

    lang_factory_name = "pytt"

    @staticmethod
    def install_extensions():
        tok2vec_attrs = [
            "pytt_last_hidden_state",
            "pytt_pooler_output",
            "pytt_all_hidden_states",
            "pytt_all_attentions",
            "pytt_d_last_hidden_state",
            "pytt_d_pooler_output",
            "pytt_d_all_hidden_states",
            "pytt_d_all_attentions",
        ]
        for attr in tok2vec_attrs:
            Doc.set_extension(attr, default=None)
            Span.set_extension(attr, getter=get_span_tok2vec_getter(attr))
            Token.set_extension(attr, getter=get_token_tok2vec_getter(attr))
        wp_attrs = ["pytt_alignment", "pytt_word_pieces", "pytt_word_pieces_"]
        for attr in wp_attrs:
            Doc.set_extension(attr, default=None)
            Span.set_extension(attr, getter=get_span_wp_getter(attr))
            Token.set_extension(attr, getter=get_token_wp_getter(attr))
        for cls in [Token, Span, Doc]:
            cls.set_extension("pytt_start", getter=get_wp_start)
            cls.set_extension("pytt_end", getter=get_wp_end)

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
        components = ["sentencizer", "pytt_wordpiecer"]
        with self.disable_pipes(*[p for p in self.pipe_names if p not in components]):
            docs = [self(doc) if isinstance(doc, str) else doc for doc in docs]
        tok2vec = self.get_pipe("pytt_tok2vec")
        pytt_outputs, backprop_tok2vec = tok2vec.begin_update(
            docs, drop=drop, **component_cfg.get("pytt_tok2vec", {})
        )
        assert len(docs) == len(pytt_outputs)
        tok2vec.set_annotations(docs, pytt_outputs)
        for doc in docs:
            assert doc._.pytt_last_hidden_state is not None
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


def get_wp_start(span):
    wp_start = span[0]._.pytt_alignment[0]
    wordpieces = span.doc._.pytt_word_pieces_
    if wp_start >= 1 and is_special_token(wordpieces[wp_start - 1]):
        return wp_start - 1
    else:
        return wp_start


def get_wp_end(span):
    wp_end = span[-1]._.pytt_alignment[-1]
    wordpieces = span.doc._.pytt_word_pieces_
    if wp_end < len(wordpieces) and is_special_token(wordpieces[wp_end + 1]):
        return wp_end + 1
    else:
        return wp_end


def get_span_wp_getter(attr):
    def span_getter(span):
        return [token._.get(attr) for token in span]

    return span_getter


def get_token_wp_getter(attr):
    def token_getter(token):
        doc_values = token.doc._.get(attr)
        return doc_values[token.i] if doc_values is not None else None

    return token_getter


def get_span_tok2vec_getter(attr):
    def span_getter(span):
        doc_activations = span.doc._.get(attr)
        if doc_activations is None:
            return None
        wp_start = span[0]._.pytt_alignment[0]
        wp_end = span[-1]._.pytt_alignment[-1]
        return doc_activations[wp_start:wp_end]

    return span_getter


def get_token_tok2vec_getter(attr):
    def token_getter(token):
        return token.doc[token.i : token.i + 1]._.get(attr)

    return token_getter
