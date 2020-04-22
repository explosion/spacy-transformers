from typing import Tuple, Callable, List, Optional

from spacy.util import registry
from thinc.api import PyTorchWrapper, Model, torch2xp, xp2torch
from .types import TransformerOutput, TokensPlus


@thinc.registry.layers("spacy.SentenceSlicer.v1")
def SentenceSlicer():
    return Model("sentence-slicer", get_sents)


def get_sents(model, docs, is_train):
    sents = []
    for doc in docs:
        sents.extend(sent.text for sent in doc.sents)
    
    def backprop(d_sents):
        return docs

    return sents, backprop


@thinc.registry.layers("spacy.TransformerTokenizer.v1")
def TransformerTokenizer(name: str) -> Model[List[List[str]], TokensPlus]:
    return Model(
        "transformer_tokenizer",
        transformer_tokenizer_forward,
        attrs={"tokenizer": AutoTokenizer.from_pretrained(name)},
    )


def transformer_tokenizer_forward(
        model, texts: List[List[str]], is_train: bool
    ) -> Tuple[TokensPlus, Callable]:
        tokenizer = model.attrs["tokenizer"]
        token_data = tokenizer.batch_encode_plus(
            [(text, None) for text in texts],
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_masks=True,
            return_input_lengths=True,
            return_tensors="pt",
        )
        return TokensPlus(**token_data), lambda d_tokens: []



@registry.architectures.register("Transformer.v1")
def TransformerModel(transformer) -> Model[TokensPlus, TransformerOutput]:
    return PyTorchWrapper(
        transformer, # e.g. via AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_inputs,
    )


def convert_transformer_inputs(model, tokens: TokensPlus, is_train):
    kwargs = {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "token_type_ids": tokens.token_type_ids,
    }
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


@thinc.registry.layers("spacy.Transformer.v1")
def Transformer(
    tokenizer: Model,
    transformer: Model,
    aligner: Model,
    slicer: Model,
) -> Model[List[Doc], AlignedTransformerOutput]:
    return Model(
        "transformer",
        transformer_forward,
        layers=[slicer, tokenizer, transformer, aligner],
        refs={
            "slicer": slicer,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "aligner": aligner
        },
    )


def transformer_forward(model: Model, docs: List[Doc], is_train: bool) -> AlignedTransformerOutput:
    # We actually could implement this pretty easily with higher-order functions:
    #   with Model.define_operators({"&": tuplify, ">>": chain}):
    #       model = (
    #            slicer
    #            >> (noop() & tokenizer)
    #            >> (getitem(0) & getitem(1) & (getitem(2) >> transformer))
    #            >> aligner
    #       )
    # That's probably how I'd do it left to my own devices, but I accept that the
    # world isn't quite ready for that sort of energy yet.
    slicer = model.get_ref("slicer")
    tokenizer = model.get_ref("tokenizer")
    transformer = model.get_ref("transformer")
    aligner = model.get_ref("aligner")

    slices, bp_slicer = slicer(docs, is_train)
    tokens, bp_tokens = tokenizer(slices, is_train)
    unaligned, bp_unaligned = transformer(tokens, is_train)
    aligned, bp_aligned = aligner((docs, slices, unaligned), is_train)

    def backprop_sentence_transformer(d_aligned):
        _, d_unaligned = bp_aligned(d_aligned)
        d_tokens = bp_unaligned(d_unaligned)
        d_slices = bp_tokens(d_tokens)
        d_docs = bp_slices(d_slices)
        return d_docs

    return aligned
