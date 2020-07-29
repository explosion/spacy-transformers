from typing import List, Callable, Iterable, Iterator, Optional, Dict, Tuple, Union
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.pipeline.pipe import deserialize_config
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.gold import Example
from spacy import util
from spacy.util import minibatch, link_vectors_to_models
from thinc.api import Model, Config, set_dropout_rate, Optimizer
import srsly
import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path

from .util import huggingface_from_pretrained, batch_by_length
from .annotation_setters import null_annotation_setter
from .data_classes import FullTransformerBatch, TransformerData
from .layers import TransformerListener


DEFAULT_CONFIG_STR = """
[transformer]
max_batch_items = 4096

[transformer.annotation_setter]
@annotation_setters = "spacy-transformer.null_annotation_setter.v1"

[transformer.model]
@architectures = "spacy-transformers.TransformerModel.v1"
name = "roberta-base"
tokenizer_config = {"use_fast": true}

[transformer.model.get_spans]
@span_getters = "strided_spans.v1"
window = 128
stride = 96
"""

DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)
DOC_EXT_ATTR = "trf_data"


@Language.factory(
    "transformer",
    assigns=[f"doc._.{DOC_EXT_ATTR}"],
    default_config=DEFAULT_CONFIG["transformer"],
)
def make_transformer(
    nlp: Language,
    name: str,
    model: Model,
    annotation_setter: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
):
    return Transformer(
        nlp.vocab, model, annotation_setter, max_batch_items=max_batch_items, name=name
    )


def install_extensions() -> None:
    if not Doc.has_extension(DOC_EXT_ATTR):
        Doc.set_extension(DOC_EXT_ATTR, default=TransformerData.empty())


class Transformer(Pipe):
    """spaCy pipeline component to use transformer models.

    The component assigns the output of the transformer to the Doc's
    extension attributes. We also calculate an alignment between the word-piece
    tokens and the spaCy tokenization, so that we can use the last hidden states
    to set the doc.tensor attribute. When multiple word-piece tokens align to
    the same spaCy token, the spaCy token receives the sum of their values.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model[List[Doc], FullTransformerBatch],
        annotation_setter: Callable = null_annotation_setter,
        *,
        name: str = "transformer",
        max_batch_items: int = 128 * 32,  # Max size of padded batch
    ):
        """Initialize the transformer component."""
        self.name = name
        self.vocab = vocab
        self.model = model
        if not isinstance(self.model, Model):
            raise ValueError(f"Expected Thinc Model, got: {type(self.model)}")
        self.annotation_setter = annotation_setter
        self.cfg = {"max_batch_items": max_batch_items}
        self.listeners: List[TransformerListener] = []
        install_extensions()

    def create_listener(self) -> None:
        listener = TransformerListener(upstream_name="transformer")
        self.listeners.append(listener)

    def add_listener(self, listener: TransformerListener) -> None:
        self.listeners.append(listener)

    def find_listeners(self, model: Model) -> None:
        for node in model.walk():
            if (
                isinstance(node, TransformerListener)
                and node.upstream_name == self.name
            ):
                self.add_listener(node)

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to one document. The document is modified in place,
        and returned. This usually happens under the hood when the nlp object
        is called on a text and all components are applied to the Doc.

        docs (Doc): The Doc to preocess.
        RETURNS (Doc): The processed Doc.

        DOCS: https://spacy.io/api/transformer#call
        """
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.

        DOCS: https://spacy.io/api/transformer#pipe
        """
        for outer_batch in minibatch(stream, batch_size):
            outer_batch = list(outer_batch)
            for indices in batch_by_length(outer_batch, self.cfg["max_batch_items"]):
                subbatch = [outer_batch[i] for i in indices]
                self.set_annotations(subbatch, self.predict(subbatch))
            yield from outer_batch

    def predict(self, docs: Iterable[Doc]) -> FullTransformerBatch:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns the extracted features as the FullTransformerBatch dataclass.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS (FullTransformerBatch): The extracted features.

        DOCS: https://spacy.io/api/transformer#predict
        """
        activations = self.model.predict(docs)
        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners:
            listener.receive(batch_id, activations.doc_data, None)
        return activations

    def set_annotations(
        self, docs: Iterable[Doc], predictions: FullTransformerBatch
    ) -> None:
        """Assign the extracted features to the Doc objects and overwrite the
        vector and similarity hooks.

        docs (Iterable[Doc]): The documents to modify.
        predictions: (FullTransformerBatch): A batch of activations.

        DOCS: https://spacy.io/api/pipe#set_annotations
        """
        doc_data = list(predictions.doc_data)
        for doc, data in zip(docs, doc_data):
            doc._.trf_data = data
        self.annotation_setter(docs, predictions)

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
        set_annotations: bool = False,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model.

        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        set_annotations (bool): Whether or not to update the Example objects
            with the predictions.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.

        DOCS: https://spacy.io/api/transformer#update
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
        return losses

    def get_loss(self, docs, golds, scores):
        pass

    def begin_training(
        self,
        get_examples: Callable[[], Iterable[Example]] = lambda: [],
        *,
        pipeline: Optional[List[Tuple[str, Callable[[Doc], Doc]]]] = None,
        sgd: Optional[Optimizer] = None,
    ):
        """Initialize the pipe for training, using data examples if available.

        get_examples (Callable[[], Iterable[Example]]): Optional function that
            returns gold-standard Example objects.
        pipeline (List[Tuple[str, Callable]]): Optional list of pipeline
            components that this component is part of. Corresponds to
            nlp.pipeline.
        sgd (thinc.api.Optimizer): Optional optimizer. Will be created with
            create_optimizer if it doesn't exist.
        RETURNS (thinc.api.Optimizer): The optimizer.

        DOCS: https://spacy.io/api/transformer#begin_training
        """
        docs = [Doc(Vocab(), words=["hello"])]
        self.model.initialize(X=docs)
        link_vectors_to_models(self.vocab)

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> None:
        """Serialize the pipe to disk.

        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.

        DOCS: https://spacy.io/api/transformer#to_disk
        """

        def save_model(p):
            trf_dir = Path(p).absolute()
            if not trf_dir.exists():
                trf_dir.mkdir()
            self.model.attrs["tokenizer"].save_pretrained(str(trf_dir))
            transformer = self.model.layers[0].shims[0]._model
            torch.save(transformer.state_dict(), trf_dir / WEIGHTS_NAME)
            transformer.config.to_json_file(trf_dir / CONFIG_NAME)

        serialize = {}
        serialize["cfg"] = lambda p: srsly.write_json(p, self.cfg)
        serialize["vocab"] = lambda p: self.vocab.to_disk(p)
        serialize["model"] = lambda p: save_model(p)
        util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> "Transformer":
        """Load the pipe from disk.

        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        RETURNS (Transformer): The loaded object.

        DOCS: https://spacy.io/api/transformer#from_disk
        """

        def load_model(p):
            p = Path(p).absolute()
            tokenizer, transformer = huggingface_from_pretrained(
                p, self.model.attrs["tokenizer_config"]
            )
            self.model.attrs["tokenizer"] = tokenizer
            self.model.attrs["set_transformer"](self.model, transformer)

        deserialize = {
            "vocab": self.vocab.from_disk,
            "cfg": lambda p: self.cfg.update(deserialize_config(p)),
            "model": load_model,
        }
        util.from_disk(path, deserialize, exclude)
        return self
