from typing import List

import torch
from spacy.tokens import Span, Doc
import thinc
from thinc.api import PyTorchWrapper, Model, ArgsKwargs
from transformers import AutoModel, AutoTokenizer

from .types import TransformerOutput, TokensPlus


def get_doc_spans(docs: List[Doc]) -> List[Span]:
    return [doc[:] for doc in docs]


def get_sents(docs: List[Doc]) -> List[Span]:
    spans = []
    for doc in docs:
        spans.extend(doc.sents)
    return spans
