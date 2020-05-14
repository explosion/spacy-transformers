from typing import List, Tuple, Callable, Union
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
from spacy.tokens import Doc
from thinc.api import PyTorchWrapper, Model
from thinc.types import ArgsKwargs
from spacy.util import registry
from pathlib import Path

from .util import huggingface_tokenize
from .util import BatchEncoding, FullTransformerBatch, TransformerData
from ._align import get_alignment


class TRFModel(Model):

    def to_disk(self, path: Union[Path, str]) -> None:
        """Serialize the model to disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        print()
        print("TRFModel to_disk")
        path.mkdir()
        print("attrs", self.attrs)
        model_path = Path(path) / "model"
        with model_path.open("wb") as file_:
            file_.write(self.to_bytes())

        # save tokenizer
        tokenizer = self.attrs["tokenizer"]
        print("tokenizer to disk:", type(tokenizer))
        tokenizer.save_pretrained(str(path))

        # save get_spans
        # self.attrs["get_spans"]) : config

        # save Transformer model
        transformer = self.layers[0].shims[0]._model
        output_model_file = Path(path) / WEIGHTS_NAME
        output_config_file = Path(path) / CONFIG_NAME
        torch.save(transformer.state_dict(), output_model_file)
        transformer.config.to_json_file(output_config_file)

    def from_disk(self, path: Union[Path, str]) -> "Model":
        """Deserialize the model from disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        print()
        print("TRFModel from_disk")
        model_path = Path(path) / "model"
        with model_path.open("rb") as file_:
            bytes_data = file_.read()
        self.from_bytes(bytes_data)
        print("attrs", self.attrs)


@registry.architectures.register("spacy.TransformerByName.v2")
def TransformerModelByName(
    name: str, get_spans: Callable, fast_tokenizer: bool
) -> Model[List[Doc], TransformerData]:
    transformer = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=fast_tokenizer)
    model = TransformerModel(transformer, tokenizer, get_spans=get_spans)
    return model


@registry.architectures.register("spacy.TransformerModel.v1")
def TransformerModel(
    transformer, tokenizer, get_spans: Callable
) -> Model[List[Doc], TransformerData]:
    wrapper = PyTorchTransformer(transformer)
    return TRFModel(
        "transformer",
        forward,
        layers=[wrapper],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans},
        dims={"nO": None},
    )


@registry.architectures.register("spacy.TransformerFromFile.v1")
def TransformerFromFile(
    dir: str, get_spans: Callable
) -> Model[List[Doc], TransformerData]:
    main_dir = Path(dir)
    dir = main_dir / "model"

    print("running spacy.TransformerFromFile.v1")
    print(" - dir", type(dir), str(dir))

    tokenizer = AutoTokenizer.from_pretrained(str(dir))      # BertTokenizer
    transformer = AutoModel.from_pretrained(str(dir))

    print(" - transformer", type(transformer))
    print(" - tokenizer", type(tokenizer))
    print(" - get_spans", type(get_spans))
    wrapper = PyTorchTransformer(transformer)
    return TransformerModel(transformer, tokenizer, get_spans)


def forward(
    model: Model, docs: List[Doc], is_train: bool
) -> Tuple[FullTransformerBatch, Callable]:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    spans = get_spans(docs)
    span_docs = {id(span.doc) for span in spans}
    for doc in docs:
        if id(doc) not in span_docs:
            raise ValueError(doc.text)
    token_data = huggingface_tokenize(tokenizer, [span.text for span in spans])
    tensors, bp_tensors = transformer(token_data, is_train)
    output = FullTransformerBatch(
        spans=spans,
        tokens=token_data,
        tensors=tensors,
        align=get_alignment(spans, token_data["input_texts"]),
    )

    def backprop_transformer(d_output: FullTransformerBatch) -> List[Doc]:
        _ = bp_tensors(d_output.tensors)
        return docs

    return output, backprop_transformer


def PyTorchTransformer(transformer):
    return PyTorchWrapper(
        transformer,  # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=_convert_transformer_inputs,
        convert_outputs=_convert_transformer_outputs,
    )


def _convert_transformer_inputs(model, tokens: BatchEncoding, is_train):
    # Adapter for the PyTorchWrapper. See https://thinc.ai/docs/usage-frameworks
    kwargs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }
    if "token_type_ids" in tokens:
        kwargs["token_type_ids"] = tokens["token_type_ids"]
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def _convert_transformer_outputs(model, inputs_outputs, is_train):
    _, tensors = inputs_outputs

    def backprop(d_tensors: List[torch.Tensor]) -> ArgsKwargs:
        return ArgsKwargs(args=(tensors,), kwargs={"grad_tensors": d_tensors})

    return tensors, backprop
