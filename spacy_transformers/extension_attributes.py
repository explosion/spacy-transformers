from spacy.tokens import Doc, Span, Token
from .util import ATTRS


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
