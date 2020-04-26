from spacy.tokens import Token, Span, Doc
from .types import TransformerOutput


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerOutput.empty())
    Span.set_extension("trf_row", default=-1)
    Token.set_extension("trf_alignment", default=[])


def get_doc_spans(docs):
    return [doc[:] for doc in docs]

def get_sent_spans(docs):
    sents = []
    for doc in docs:
        sents.extend(doc.sents)
    return sents
