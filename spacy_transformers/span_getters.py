from typing import Callable, Iterable, List
from spacy.tokens import Doc, Span

from .util import registry


@registry.span_getters("strided_spans.v1")
def configure_strided_spans(window: int, stride: int) -> Callable:
    def get_strided_spans(docs: Iterable[Doc]) -> List[Span]:
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


@registry.span_getters("sent_spans.v1")
def configure_get_sent_spans() -> Callable:
    def get_sent_spans(docs: Iterable[Doc]) -> List[Span]:
        return [list(doc.sents) for doc in docs]

    return get_sent_spans


@registry.span_getters("doc_spans.v1")
def configure_get_doc_spans() -> Callable:
    def get_doc_spans(docs: Iterable[Doc]) -> List[Span]:
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
