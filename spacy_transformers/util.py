from spacy.tokens import Token, Span, Doc


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerOutput.empty())
    Span.set_extension("trf_row", default=-1)
    Token.set_extension("trf_alignment", default=[])
    #Doc.set_extension("trf_get_features", method=get_doc_features)
    #Doc.set_extension("trf_get_features_1d", method=get_doc_features_1d)
    #Doc.set_extension("trf_get_features_2d", method=get_doc_features_2d)
    #Doc.set_extension("trf_get_features_3d", method=get_doc_features_3d)
    #Span.set_extension("trf_get_features", method=get_span_features)
    #Span.set_extension("trf_get_features_1d", method=get_span_features_1d)
    #Span.set_extension("trf_get_features_2d", method=get_span_features_2d)
    #Span.set_extension("trf_get_features_3d", method=get_span_features_3d)
    #Token.set_extension("trf_get_features", method=get_token_features)
    #Token.set_extension("trf_get_features_1d", method=get_token_features_1d)
    #Token.set_extension("trf_get_features_2d", method=get_token_features_2d)
    #Token.set_extension("trf_get_features_3d", method=get_token_features_3d)
