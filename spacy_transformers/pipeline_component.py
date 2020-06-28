from typing import List, Callable
from spacy.pipeline import Pipe
from spacy.language import component
from spacy.pipeline.pipes import _load_cfg
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.gold import Example
from spacy import util
from spacy.util import minibatch, link_vectors_to_models
from spacy_transformers.util import huggingface_from_pretrained
from thinc.api import Model, set_dropout_rate

import srsly
import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path

from .annotation_setters import null_annotation_setter
from .data_classes import FullTransformerBatch, TransformerData
from .layers import TransformerListener


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerData.empty(), force=True)


@component("transformer", assigns=["doc._.trf_data"])
class Transformer(Pipe):
    """spaCy pipeline component to use transformer models.

    The component assigns the output of the transformer to the Doc's
    extension attributes. We also calculate an alignment between the word-piece
    tokens and the spaCy tokenization, so that we can use the last hidden states
    to set the doc.tensor attribute. When multiple word-piece tokens align to
    the same spaCy token, the spaCy token receives the sum of their values.
    """

    @classmethod
    def from_nlp(cls, nlp, model, **cfg):
        return cls(nlp.vocab, model, **cfg)

    def __init__(
        self,
        vocab: Vocab,
        model: Model[List[Doc], FullTransformerBatch],
        annotation_setter: Callable = null_annotation_setter,
        *,
        max_batch_size: int = 8,
        **cfg,
    ):
        self.vocab = vocab
        self.model = model
        assert isinstance(self.model, Model)
        self.annotation_setter = annotation_setter
        self.cfg = dict(cfg)
        self.cfg["max_batch_size"] = max_batch_size
        self.listeners: List[TransformerListener] = []
        install_extensions()

    def create_listener(self):
        listener = TransformerListener(upstream_name="transformer")
        self.listeners.append(listener)

    def add_listener(self, listener):
        self.listeners.append(listener)

    def find_listeners(self, model):
        for node in model.walk():
            if (
                isinstance(node, TransformerListener)
                and node.upstream_name == self.name
            ):
                self.add_listener(node)

    def __call__(self, doc):
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream, batch_size=128, n_threads=-1, as_example=False):
        batch_size = min(batch_size, self.cfg["max_batch_size"])
        for batch in minibatch(stream, batch_size):
            batch = list(batch)
            # The model is currently a bit brittle if there's duplicate docs
            seen_ids = set()
            uniqued = []
            for doc in batch:
                if id(doc) not in seen_ids:
                    uniqued.append(doc)
                    seen_ids.add(id(doc))
            outputs = self.predict(uniqued)
            self.set_annotations(uniqued, outputs)
            yield from batch

    def predict(self, docs) -> FullTransformerBatch:
        activations = self.model.predict(docs)
        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners:
            listener.receive(batch_id, activations.doc_data, None)
        return activations

    def set_annotations(self, docs: List[Doc], predictions: FullTransformerBatch):
        """Assign the extracted features to the Doc objects and overwrite the
        vector and similarity hooks.

        docs (iterable): A batch of `Doc` objects.
        activations (iterable): A batch of activations.
        """
        doc_data = list(predictions.doc_data)
        assert len(docs) == len(doc_data), (len(docs), len(doc_data))
        for doc, data in zip(docs, doc_data):
            doc._.trf_data = data
        self.annotation_setter(docs, predictions)

    def update(self, examples, drop=0.0, sgd=None, losses=None, set_annotations=False):
        """Update the model.
        examples (iterable): A batch of examples
        drop (float): The droput rate.
        sgd (callable): An optimizer.
        RETURNS (dict): Results from the update.
        """
        if losses is None:
            losses = {}
        docs = [eg.predicted for eg in examples]
        if isinstance(docs, Doc):
            docs = [docs]
        set_dropout_rate(self.model, drop)
        trf_full, bp_trf_full = self.model.begin_update(docs)
        d_tensors = []

        losses.setdefault(self.name, 0.0)

        def accumulate_gradient(d_trf_datas: List[TransformerData]):
            """Accumulate tok2vec loss and gradient. This is passed as a callback
            to all but the last listener. Only the last one does the backprop.
            """
            nonlocal d_tensors
            for i, d_trf_data in enumerate(d_trf_datas):
                for d_tensor in d_trf_data.tensors:
                    losses[self.name] += float((d_tensor ** 2).sum())  # type: ignore
                if i >= len(d_tensors):
                    d_tensors.append(d_trf_data.tensors)
                else:
                    for j, d_tensor in enumerate(d_trf_data.tensors):
                        d_tensors[i][j] += d_tensor

        def backprop(d_trf_datas: List[TransformerData]):
            """Callback to actually do the backprop. Passed to last listener."""
            nonlocal d_tensors
            accumulate_gradient(d_trf_datas)
            d_trf_full = trf_full.unsplit_by_doc(d_tensors)
            d_docs = bp_trf_full(d_trf_full)
            if sgd is not None:
                self.model.finish_update(sgd)
            d_tensors = []
            return d_docs

        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners[:-1]:
            listener.receive(batch_id, trf_full.doc_data, accumulate_gradient)
        self.listeners[-1].receive(batch_id, trf_full.doc_data, backprop)
        if set_annotations:
            self.set_annotations(docs, trf_full)

    def get_loss(self, docs, golds, scores):
        pass

    def begin_training(
        self, get_examples=lambda: [], pipeline=None, sgd=None, **kwargs
    ):
        """Allocate models and pre-process training data

        get_examples (function): Function returning example training data.
        pipeline (list): The pipeline the model is part of.
        """
        docs = [Doc(Vocab(), words=["hello"])]
        self.model.initialize(X=docs)
        link_vectors_to_models(self.vocab)

    def to_disk(self, path, exclude=tuple(), **kwargs):
        """Serialize the pipe and its model to disk."""

        def save_model(p):
            trf_dir = Path(p).absolute()
            trf_dir.mkdir()
            self.model.attrs["tokenizer"].save_pretrained(str(trf_dir))
            transformer = self.model.layers[0].shims[0]._model
            torch.save(transformer.state_dict(), trf_dir / WEIGHTS_NAME)
            transformer.config.to_json_file(trf_dir / CONFIG_NAME)

        serialize = {}
        serialize["cfg"] = lambda p: srsly.write_json(p, self.cfg)
        serialize["vocab"] = lambda p: self.vocab.to_disk(p)
        serialize["model"] = lambda p: save_model(p)
        exclude = util.get_serialization_exclude(serialize, exclude, kwargs)
        util.to_disk(path, serialize, exclude)

    def from_disk(self, path, exclude=tuple(), **kwargs):
        """Load the pipe and its model from disk."""

        def load_model(p):
            p = Path(p).absolute()
            tokenizer, transformer = huggingface_from_pretrained(
                p, self.model.attrs["tokenizer_config"]
            )
            self.model.attrs["tokenizer"] = tokenizer
            self.model.attrs["set_transformer"](self.model, transformer)

        deserialize = {
            "vocab": self.vocab.from_disk,
            "cfg": lambda p: self.cfg.update(_load_cfg(p)),
            "model": load_model,
        }
        exclude = util.get_serialization_exclude(deserialize, exclude, kwargs)
        util.from_disk(path, deserialize, exclude)
        return self
