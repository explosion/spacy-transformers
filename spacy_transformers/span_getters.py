from typing import Callable, Iterable, List
from spacy.tokens import Doc, Span

from .util import registry

SpannerT = Callable[[List[Doc]], List[List[Span]]]


@registry.span_getters("spacy-transformers.strided_spans.v1")
def configure_strided_spans(window: int, stride: int) -> SpannerT:
    """
    Set the 'window' and 'stride' options for getting strided spans.

    If you set the window and stride to the same value, the spans will cover
    each token once. Setting 'stride' lower than 'window' will allow for an
    overlap, so that some tokens are counted twice. This can be desirable, 
    because it allows all tokens to have both a left and right context.
    """

    def get_strided_spans(docs: Iterable[Doc]) -> List[List[Span]]:
        spans = []
        for doc in docs:
            start = 0
            spans.append([])
            for i in range(len(doc) // stride):
                spans[-1].append(doc[start : start + window])
                if (start + window) >= len(doc):
                    break
                start += stride
            else:
                if start < len(doc):
                    spans[-1].append(doc[start:])
        return spans

    return get_strided_spans


@registry.span_getters("spacy-transformers.sent_spans.v1")
def configure_get_sent_spans() -> Callable:
    """
    Create a `span_getter` that uses sentence boundary markers to extract
    the spans. This requires sentence boundaries to be set, and may result
    in somewhat uneven batches, depending on the sentence lengths. However,
    it does provide the transformer with more meaningful windows to attend over.
    """

    def get_sent_spans(docs: Iterable[Doc]) -> List[List[Span]]:
        return [list(doc.sents) for doc in docs]

    return get_sent_spans


@registry.span_getters("spacy-transformers.doc_spans.v1")
def configure_get_doc_spans() -> Callable:
    """
    Create a `span_getter` that uses the whole document as its spans. This is
    the best approach if your `Doc` objects already refer to relatively short
    texts.
    """

    def get_doc_spans(docs: Iterable[Doc]) -> List[List[Span]]:
        return [[doc[:]] for doc in docs]

    return get_doc_spans


get_sent_spans = configure_get_sent_spans()
get_doc_spans = configure_get_doc_spans()


__all__ = [
    "get_sent_spans",
    "get_doc_spans",
    "configure_get_doc_spans",
    "configure_get_sent_spans",
    "configure_strided_spans",
]
