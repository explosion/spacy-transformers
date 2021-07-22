from typing import Callable, List
from spacy.tokens import Doc

from .util import registry
from .data_classes import FullTransformerBatch


def null_annotation_setter(docs: List[Doc], trf_data: FullTransformerBatch) -> None:
    """Set no additional annotations on the Doc objects."""
    pass


@registry.annotation_setters("spacy-transformers.null_annotation_setter.v1")
def configure_null_annotation_setter() -> Callable[
    [List[Doc], FullTransformerBatch], None
]:
    return null_annotation_setter


null_annotation_setter = configure_null_annotation_setter()


__all__ = ["null_annotation_setter", "configure_null_annotation_setter"]
