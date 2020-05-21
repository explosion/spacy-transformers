from typing import Callable
from .util import registry


@registry.annotation_setters("spacy-transformers.strided_spans.v1")
def configure_strided_spans(window: int, stride: int) -> Callable:
    def get_strided_spans(docs):
        spans = []
        for doc in docs:
            start = 0
            for i in range(len(doc) // stride):
                spans.append(doc[start : start + window])
                if (start + window) >= len(doc):
                    break
                start += stride
            else:
                if start < len(doc):
                    spans.append(doc[start:])
        return spans

    return get_strided_spans


@registry.annotation_setters("spacy-transformers.get_sent_spans.v1")
def configure_get_sent_spans():
    def get_sent_spans(docs):
        sents = []
        for doc in docs:
            sents.extend(doc.sents)
        return sents

    return get_sent_spans


@registry.annotation_setters("spacy-transformers.get_doc_spans.v1")
def configure_get_doc_spans():
    def get_doc_spans(docs):
        return [doc[:] for doc in docs]

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
