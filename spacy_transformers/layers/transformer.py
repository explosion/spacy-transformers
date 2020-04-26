@thinc.registry.layers("spacy.TransformerByName.v1")
def TransformerByName(
    name: str, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    transformer = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return Transformer(transformer, tokenizer, get_spans=get_spans)


@thinc.registry.layers("spacy.Transformer.v1")
def Transformer(
    transformer, tokenizer, get_spans=get_doc_spans
) -> Model[List[Doc], TransformerOutput]:
    wrapper = PyTorchTransformer(transformer)
    return Model(
        "transformer",
        forward,
        layers=[transformer],
        attrs={"tokenizer": tokenizer, "get_spans": get_spans},
        dims={"nO": wrapper.get_dim("nO")}
    )


def forward(
    model: Model, docs: List[Doc], is_train: bool
) -> TransformerOutput:
    tokenizer = model.attrs["tokenizer"]
    get_spans = model.attrs["get_spans"]
    transformer = model.layers[0]

    spans = get_spans(docs)

    token_data = tokenizer.batch_encode_plus(
        [span.text for span in spans],
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors=None,  # Work around bug :(
        return_token_type_ids=None,  # Sets to model default
    )
    # Work around https://github.com/huggingface/transformers/issues/3224
    token_data["input_ids"] = torch.tensor(token_data["input_ids"])
    token_data["attention_mask"] = torch.tensor(token_data["attention_mask"])
    if "token_type_ids" in token_data:
        token_data["token_type_ids"] = torch.tensor(token_data["token_type_ids"])
    tokens = TokensPlus(**token_data)

    tensors, bp_tensors = transformer(tokens, is_train)
    output = TransformerOutput(
        tokens=tokens, tensors=tensors, spans=spans, ops=transformer.ops
    )

    def backprop_transformer(d_output: TransformerOutput):
        _ = bp_tensors(d_output.tensors)
        return docs

    return output, backprop_transformer
