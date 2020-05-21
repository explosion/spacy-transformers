@registry.architectures.register("spacy-transformers.listener.v1")
def transformer_listener_tok2vec_v1(
    pooling, width: int, grad_factor: float = 1.0
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        TransformerListener("transformer", width=width),
        trf_data_to_tensor(pooling, width, grad_factor),
    )


@registry.architectures.register("spacy.Tok2VecTransformer.v1")
def transformer_tok2vec_v1(
    transformer,
    pooling,
    get_spans,
    grad_factor: float = 1.0,
) -> Model[List[TransformerData], List[Floats2d]]:
    return chain(
        transformer,
        get_trf_data(),
        trf_data_to_tensor(pooling, width, grad_factor)
    )
