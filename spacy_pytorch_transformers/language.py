from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class
from spacy.gold import GoldParse

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
        sentencizer = self.get_pipe("sentencizer")
        wp = self.get_pipe("pytt_wordpiecer")
        tok2vec = self.get_pipe("pytt_tok2vec")
        new_docs = []
        new_golds = []
        for doc, gold in zip(docs, golds):
            if isinstance(doc, str):
                doc = self.make_doc(doc)
            doc = sentencizer(doc)
            doc = wp(doc)
            if not isinstance(gold, GoldParse):
                gold = GoldParse(doc, **gold)
            new_docs.append(doc)
            new_golds.append(gold)
        docs = new_docs
        golds = new_golds
        pytt_outputs, backprop_tok2vec = tok2vec.begin_update(
            docs, drop=drop, **component_cfg.get("pytt_tok2vec", {})
        )
        tok2vec.set_annotations(docs, pytt_outputs)
        for doc in docs:
            assert doc._.pytt_last_hidden_state is not None
        with self.disable_pipes("pytt_tok2vec"):
            super().update(
                docs,
                golds,
                drop=0.0,
                sgd=sgd,
                losses=losses,
                component_cfg=component_cfg,
            )
        backprop_tok2vec(docs, sgd=sgd)

    def resume_training(self, sgd=None, component_cfg=None, **kwargs):
        """Continue training a pre-trained model.

        Before running the normal Language.resume_training method, we do the
        following:

        * Look for a tok2vec pipeline component. By default we look for the name
            'pytt_tok2vec'. This can be changed with the tok2vec_name keyword
            argument. If no component is found, a ValueError is raised.
        * If any other components have `component.model == True` and a `.begin_training()`
            method, we call the `.begin_training()` method. Configuration can
            be passed in using the component_cfg keyword argument. If unset,
            we also pass in a value for token_vector_width, which we read from
            the tok2vec component.
        """
        if component_cfg is None:
            component_cfg = {}
        tok2vec_name = kwargs.get("tok2vec_name", "pytt_tok2vec")
        tok2vec = self.get_pipe(tok2vec_name)
        token_vector_width = tok2vec.token_vector_width
        for name, component in self.pipeline:
            if name == tok2vec_name:
                continue
            elif getattr(component, "model", None) is not True:
                continue
            elif not hasattr(component, "begin_training"):
                continue
            cfg = component_cfg.get(name, {})
            if "tok2vec_name" not in component_cfg:
                cfg["tok2vec_name"] = tok2vec_name
            if "token_vector_width" not in component_cfg:
                cfg["token_vector_width"] = token_vector_width
            component.cfg.update(cfg)
            component.begin_training(pipeline=self.pipeline, sgd=False, **cfg)
            assert component.model is not True
        optimizer = super().resume_training(sgd=sgd, **kwargs)
        optimizer.L2 = 0.0
        return optimizer


def get_defaults(lang):
    """Get the language-specific defaults, if available in spaCy."""
    try:
        lang_cls = get_lang_class(lang)
        return lang_cls.Defaults
    except ImportError:
        return Language.Defaults


def get_wp_start(span):
    if isinstance(span, Token):
        span = span.doc[span.i : span.i + 1]
    for token in span:
        if token._.pytt_alignment:
            wp_start = token._.pytt_alignment[0]
            break
    else:
        return None
    wordpieces = span.doc._.pytt_word_pieces_
    if wp_start >= 1 and is_special_token(wordpieces[wp_start - 1]):
        return wp_start - 1
    else:
        return wp_start


def get_wp_end(span):
    if isinstance(span, Token):
        span = span.doc[span.i : span.i + 1]
    for token in reversed(span):
        if token._.pytt_alignment:
            wp_end = token._.pytt_alignment[-1]
            break
    else:
        return None
    wordpieces = span.doc._.pytt_word_pieces_
    next_token = wp_end + 1
    if next_token < len(wordpieces) and is_special_token(wordpieces[next_token]):
        return next_token
    else:
        return wp_end


def get_span_wp_getter(attr):
    def span_alignment_getter(span):
        return [token._.get(attr) for token in span]

    def span_getter(span):
        start = span._.pytt_start
        end = span._.pytt_end
        if start is None and end is None:
            return []
        doc_values = span.doc._.get(attr)
        start = start if start is not None else 0
        if end is None:
            return doc_values[start:]
        return doc_values[start : end + 1]

    if attr == "pytt_alignment":
        return span_alignment_getter
    else:
        return span_getter


def get_token_wp_getter(attr):
    def token_alignment_getter(token):
        doc_values = token.doc._.get(attr)
        return doc_values[token.i] if doc_values is not None else None

    def token_wordpiece_getter(token):
        doc_values = token.doc._.get(attr)
        start = token._.pytt_start
        end = token._.pytt_end
        if start is None and end is None:
            return []
        return [doc_values[i] for i in range(start, end + 1)]

    if attr == "pytt_alignment":
        return token_alignment_getter
    else:
        return token_wordpiece_getter


def get_span_tok2vec_getter(attr):
    def span_getter(span):
        doc_activations = span.doc._.get(attr)
        if doc_activations is None:
            return None
        wp_start = span[0]._.pytt_wp_start
        wp_end = span[-1]._.pytt_wp_end
        if wp_start is not None and wp_end is not None:
            return doc_activations[wp_start : wp_end + 1]
        else:
            # Return empty slice.
            return doc_activations[0:0]

    return span_getter


def get_token_tok2vec_getter(attr):
    def token_getter(token):
        # Delegate through span, so get a span with just the token.
        span = token.doc[token.i : token.i + 1]
        return span._.get(attr)

    return token_getter
