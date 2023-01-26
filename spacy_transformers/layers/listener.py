from typing import Optional, Callable, List
from thinc.api import Model
from spacy.errors import Errors
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
        if self._backprop is not None:
            result = self._backprop(*args, **kwargs)
        else:
            result = None
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
        # This might occur during training when the transformer layer is frozen / hasn't been updated.
        # In that case, it should be set to "annotating" so we can retrieve the embeddings from the doc.
        if model._batch_id is None:
            outputs = []
            for doc in docs:
                if doc._.trf_data is None:
                    raise ValueError(Errors.E203.format(name="transformer"))
                else:
                    outputs.append(doc._.trf_data)
            return outputs, _empty_backprop
        else:
            model.verify_inputs(docs)
            return model._outputs, model.backprop_and_clear
    else:
        width = model.get_dim("nO")
        outputs = []
        for doc in docs:
            if doc._.trf_data is None:
                outputs.append(TransformerData.zeros(len(doc), width, xp=model.ops.xp))
            else:
                outputs.append(doc._.trf_data)
        return outputs, _empty_backprop


def _empty_backprop(dX):
    return []
