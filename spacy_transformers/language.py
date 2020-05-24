from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import get_lang_class
from spacy.gold import GoldParse
from .util import is_special_token, pkg_meta, ATTRS, PIPES, LANG_FACTORY


class TransformersLanguage(Language):
    """A subclass of spacy.Language that holds a Transformer pipeline.

    Transformer pipelines work only slightly differently from spaCy's default
    pipelines. Specifically, we introduce a new pipeline component at the start
    of the pipeline, TransformerTok2Vec. We then modify the nlp.update()
    function to run the TransformerTok2Vec before the other pipeline components,
    and backprop it after the other components are done.
    """

    lang_factory_name = LANG_FACTORY

    @staticmethod
    def install_extensions():
        tok2vec_attrs = [
            ATTRS.last_hidden_state,
            ATTRS.pooler_output,
            ATTRS.all_hidden_states,
            ATTRS.all_attentions,
            ATTRS.d_last_hidden_state,
            ATTRS.d_pooler_output,
            ATTRS.d_all_hidden_states,
            ATTRS.d_all_attentions,
        ]
        for attr in tok2vec_attrs:
            Doc.set_extension(attr, default=None)
            Span.set_extension(attr, getter=get_span_tok2vec_getter(attr))
            Token.set_extension(attr, getter=get_token_tok2vec_getter(attr))
        wp_attrs = [ATTRS.alignment, ATTRS.word_pieces, ATTRS.word_pieces_]
        for attr in wp_attrs:
            Doc.set_extension(attr, default=None)
            Span.set_extension(attr, getter=get_span_wp_getter(attr))
            Token.set_extension(attr, getter=get_token_wp_getter(attr))
        Doc.set_extension(ATTRS.separator, default=None)
        Span.set_extension(
            ATTRS.separator, getter=lambda span: span.doc._.get(ATTRS.separator)
        )
        Token.set_extension(
            ATTRS.separator, getter=lambda token: token.doc._.get(ATTRS.separator)
        )
        Doc.set_extension(ATTRS.segments, getter=get_segments)
        Span.set_extension(ATTRS.segments, getter=get_segments)
        for cls in [Token, Span, Doc]:
            cls.set_extension(ATTRS.start, getter=get_wp_start)
            cls.set_extension(ATTRS.end, getter=get_wp_end)

    def __init__(
        self, vocab=True, make_doc=True, max_length=10 ** 6, meta={}, **kwargs
    ):
        """Initialize the language class. Expects either a trf_name setting in
        the meta or as a keyword argument, specifying the pre-trained model
        name. This is used to set up the model-specific tokenizer.
        """
        meta = dict(meta)
        meta["lang_factory"] = self.lang_factory_name
        # Add this package to requirements to it will be included in the
        # install_requires of any model using this language class
        package = f"{pkg_meta['title']}>={pkg_meta['version']}"
        meta.setdefault("requirements", []).append(package)
        self.lang = meta.get("lang", "xx")
        self.Defaults = get_defaults(self.lang)
        super().__init__(vocab, make_doc, max_length, meta=meta, **kwargs)

    def update(self, docs, golds, drop=0.0, sgd=None, losses=None, component_cfg={}):
        component_cfg = dict(component_cfg)
        if self.has_pipe("sentencizer"):
            sentencizer = self.get_pipe("sentencizer")
        else:
            sentencizer = lambda doc: doc
        if self.has_pipe(PIPES.wordpiecer):
            wp = self.get_pipe(PIPES.wordpiecer)
        else:
            wp = lambda doc: doc
        tok2vec = self.get_pipe(PIPES.tok2vec)
        new_docs = []
        new_golds = []
        for doc, gold in zip(docs, golds):
            if isinstance(doc, str):
                doc = self.make_doc(doc)
            doc = sentencizer(doc)
            if doc._.get(ATTRS.word_pieces) is None:
                doc = wp(doc)
            if not isinstance(gold, GoldParse):
                gold = GoldParse(doc, **gold)
            new_docs.append(doc)
            new_golds.append(gold)
        docs = new_docs
        golds = new_golds
        outputs, backprop_tok2vec = tok2vec.begin_update(
            docs, drop=drop, **component_cfg.get(PIPES.tok2vec, {})
        )
        tok2vec.set_annotations(docs, outputs)
        for doc in docs:
            assert doc._.get(ATTRS.last_hidden_state) is not None
        with self.disable_pipes(PIPES.tok2vec):
            super().update(
                docs,
                golds,
                drop=0.1,
                sgd=sgd,
                losses=losses,
                component_cfg=component_cfg,
            )
        backprop_tok2vec(docs, sgd=sgd)

    def resume_training(self, sgd=None, component_cfg=None, **kwargs):
        """Continue training a pre-trained model.

        Before running the normal Language.resume_training method, we do the
        following:

        * Look for a tok2vec pipeline component. The component name can be
            changed with the tok2vec_name keyword
            argument. If no component is found, a ValueError is raised.
        * If any other components have `component.model == True` and a
            `.begin_training()` method, we call the `.begin_training()` method.
            Configuration can be passed in using the component_cfg keyword
            argument. If unset, we also pass in a value for token_vector_width,
            which we read from the tok2vec component.
        """
        if component_cfg is None:
            component_cfg = {}
        tok2vec_name = kwargs.get("tok2vec_name", PIPES.tok2vec)
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
        if token._.get(ATTRS.alignment):
            wp_start = token._.get(ATTRS.alignment)[0]
            break
    else:
        return None
    wordpieces = span.doc._.get(ATTRS.word_pieces_)
    # This is a messy way to check for the XLNet-style pattern, where we can
    # have <sep> <cls>. In the BERT-style pattern, we have [cls] at start.
    if is_special_token(wordpieces[0]):
        if wp_start >= 1 and is_special_token(wordpieces[wp_start - 1]):
            return wp_start - 1
    return wp_start


def get_wp_end(span):
    if isinstance(span, Token):
        span = span.doc[span.i : span.i + 1]
    for token in reversed(span):
        if token._.get(ATTRS.alignment):
            wp_end = token._.get(ATTRS.alignment)[-1]
            break
    else:
        return None
    wordpieces = span.doc._.get(ATTRS.word_pieces_)
    if (wp_end + 1) < len(wordpieces) and is_special_token(wordpieces[wp_end + 1]):
        wp_end += 1
    # This is a messy way to check for the XLNet-style pattern, where we can
    # have <sep> <cls>. In the BERT-style pattern, we have [cls] at start.
    if not is_special_token(wordpieces[0]):
        if (wp_end + 1) < len(wordpieces) and is_special_token(wordpieces[wp_end + 1]):
            wp_end += 1
    return wp_end


def get_span_wp_getter(attr):
    def span_alignment_getter(span):
        return [token._.get(attr) for token in span]

    def span_getter(span):
        start = span._.get(ATTRS.start)
        end = span._.get(ATTRS.end)
        if start is None and end is None:
            return []
        doc_values = span.doc._.get(attr)
        start = start if start is not None else 0
        if end is None:
            return doc_values[start:]
        return doc_values[start : end + 1]

    if attr == ATTRS.alignment:
        return span_alignment_getter
    else:
        return span_getter


def get_token_wp_getter(attr):
    def token_alignment_getter(token):
        doc_values = token.doc._.get(attr)
        return doc_values[token.i] if doc_values is not None else None

    def token_wordpiece_getter(token):
        doc_values = token.doc._.get(attr)
        start = token._.get(ATTRS.start)
        end = token._.get(ATTRS.end)
        if start is None and end is None:
            return []
        return [doc_values[i] for i in range(start, end + 1)]

    if attr == ATTRS.alignment:
        return token_alignment_getter
    else:
        return token_wordpiece_getter


def get_span_tok2vec_getter(attr):
    def span_getter(span):
        doc_activations = span.doc._.get(attr)
        if doc_activations is None:
            return None
        wp_start = span[0]._.get(ATTRS.start)
        wp_end = span[-1]._.get(ATTRS.end)
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


def get_segments(doc):
    separator = doc._.get(ATTRS.separator)
    if separator is not None:
        start = 0
        for token in doc:
            if token.text == separator:
                yield doc[start : token.i + 1]
                start = token.i + 1
        yield doc[start:]
    else:
        yield doc[:]
