from typing import Callable, List
from spacy.tokens import Doc

from .util import registry
from .data_classes import FullTransformerBatch, TransformerData


@registry.annotation_setters("spacy-transformers.trfdata_setter.v1")
def configure_trfdata_setter() -> Callable[
    [List[Doc], FullTransformerBatch], None
]:
    attribute = "trf_data"
    if not Doc.has_extension(attribute):
        Doc.set_extension(attribute, default=TransformerData.empty())

    def trfdata_setter(docs: List[Doc], predictions: FullTransformerBatch) -> None:
        """Set the transforrmer data to the doc._.trf_data attribute.

        docs (Iterable[Doc]): The documents to modify.
        predictions (FullTransformerBatch): The data to set, produced by Transformer.predict.
        """
        doc_data = list(predictions.doc_data)
        for doc, data in zip(docs, doc_data):
            doc._.trf_data = data


__all__ = ["trfdata_setter", "configure_trfdata_setter"]
