from typing import Optional, Callable, List
from thinc.api import Model
from spacy.tokens import Doc
from ..data_classes import TransformerData


class TransformerListener(Model):
    """A layer that gets fed its answers from an upstream connection,
    for instance from a component earlier in the pipeline.
    """

    name = "transformer-listener"

    _batch_id: Optional[int]
    _outputs: Optional[List[TransformerData]]
    _backprop: Optional[Callable[[List[TransformerData]], List[Doc]]]

    def __init__(self, upstream_name: str):
        Model.__init__(self, name=self.name, forward=forward, dims={"nO": None})
        self.upstream_name = upstream_name
        self._batch_id = None
        self._outputs = None
        self._backprop = None

    @classmethod
    def get_batch_id(cls, inputs: List[Doc]):
        return sum(sum(token.orth for token in doc) for doc in inputs)

    def receive(self, batch_id, outputs, backprop):
        self._batch_id = batch_id
        self._outputs = outputs
        self._backprop = backprop

    def backprop_and_clear(self, *args, **kwargs):
        """Call the stored _backprop callback, and then
        clears it. This saves memory, as otherwise we hold onto that callback
        until the next batch.
        """
        result = self._backprop(*args, **kwargs)
        self._batch_id = None
        self._outputs = None
        self._backprop = None
        return result

    def verify_inputs(self, inputs):
        if self._batch_id is None and self._outputs is None:
            raise ValueError
        else:
            batch_id = self.get_batch_id(inputs)
            if batch_id != self._batch_id:
                raise ValueError(f"Mismatched IDs! {batch_id} vs {self._batch_id}")
            else:
                return True


def forward(model: TransformerListener, docs, is_train):
    if is_train:
        model.verify_inputs(docs)
        return model._outputs, model.backprop_and_clear
    else:
        if len(docs) == 0:
            outputs = []
        elif any(doc._.trf_data is None for doc in docs):
            width = model.get_dim("nO")
            outputs = [
                TransformerData.zeros(len(doc), width, xp=model.ops.xp) for doc in docs
            ]
        else:
            outputs = [doc._.trf_data for doc in docs]
        return outputs, lambda d_data: []
