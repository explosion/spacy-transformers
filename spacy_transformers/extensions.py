import numpy
from spacy.tokens import Token, Span, Doc
from thinc.types import Floats1d, Floats2d, Floats3d

from .types import TransformerOutput


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerOutput.empty())
    Span.set_extension("trf_row", default=-1)
    Token.set_extension("trf_alignment", default=[])
    Doc.set_extension("trf_get_features", method=get_doc_features)
    Doc.set_extension("trf_get_features_1d", method=get_doc_features_1d)
    Doc.set_extension("trf_get_features_2d", method=get_doc_features_2d)
    Doc.set_extension("trf_get_features_3d", method=get_doc_features_3d)
    Span.set_extension("trf_get_features", method=get_span_features)
    Span.set_extension("trf_get_features_1d", method=get_span_features_1d)
    Span.set_extension("trf_get_features_2d", method=get_span_features_2d)
    Span.set_extension("trf_get_features_3d", method=get_span_features_3d)
    Token.set_extension("trf_get_features", method=get_token_features)
    Token.set_extension("trf_get_features_1d", method=get_token_features_1d)
    Token.set_extension("trf_get_features_2d", method=get_token_features_2d)
    Token.set_extension("trf_get_features_3d", method=get_token_features_3d)


def get_doc_features(doc: Doc, ndim: int, **kwargs):
    if ndim == 1:
        return doc._.trf_get_features_1d(**kwargs)
    elif ndim == 2:
        return doc._.trf_get_features_2d(**kwargs)
    elif ndim == 3:
        return doc._.trf_get_features_3d(**kwargs)
    else:
        raise ValueError


def get_span_features(span: Span, ndim: int, **kwargs):
    if ndim == 1:
        return span._.trf_get_features_1d(**kwargs)
    elif ndim == 2:
        return span._.trf_get_features_2d(**kwargs)
    elif ndim == 3:
        return span._.trf_get_features_3d(**kwargs)
    else:
        raise ValueError


def get_token_features(token: Token, ndim: int, **kwargs):
    if ndim == 1:
        return token._.trf_get_features_1d(**kwargs)
    elif ndim == 2:
        return token._.trf_get_features_2d(**kwargs)
    elif ndim == 3:
        return token._.trf_get_features_3d(**kwargs)
    else:
        raise ValueError


def get_doc_features_1d(doc: Doc, **kwargs) -> Floats1d:
    if not len(doc._.trf_spans):
        return doc[:]._.trf_get_features_1d(**kwargs)
    vectors = [span._.trf_get_features_1d(**kwargs) for span in doc._.trf_spans]
    output = vectors[0]
    for v in vectors[1:]:
        output += v
    output /= len(vectors)
    return output


def get_doc_features_2d(doc: Doc, **kwargs) -> Floats2d:
    if not len(doc._.trf_spans):
        return doc[:]._.trf_get_features_2d(**kwargs)
    # We might have repeat tokens, in which case we take their average vector.
    row_freqs = doc._.trf_data.ops.alloc_2f(len(doc), 1)
    for span in doc._.trf_spans:
        row_freqs[span.start : span.end] += 1
    # Fill to 1 so we don't get nans.
    row_freqs[row_freqs == 0] = 1
    # Now get the array2d
    arrays = [span._.trf_get_features_2d(**kwargs) for span in doc._.trf_spans]
    output = tensors[0]
    for x in arrays[1:]:
        output += x
    output /= row_freqs
    return output


def get_doc_features_3d(doc: Doc, **kwargs) -> Floats3d:
    if not len(doc._.trf_spans):
        return doc[:]._.trf_get_features_3d(**kwargs)
    # We might have repeat tokens, in which case we take their average vector.
    row_freqs = doc._.trf_data.ops.alloc_3f(len(doc), 1)
    for span in doc._.trf_spans:
        row_freqs[span.start : span.end] += 1
    # Fill to 1 so we don't get nans.
    row_freqs[row_freqs == 0] = 1
    # Now get the tensor
    arrays = [span._.trf_get_features_3d(**kwargs) for span in doc._.trf_spans]
    output = arrays[0]
    for x in arrays[1:]:
        output += x
    output /= row_freqs
    return output


def get_span_features_1d(span: Span, **kwargs) -> Floats1d:
    wp_array = span.doc._.trf_data.arrays[-1][span._.trf_row]
    return wp_array.mean(axis=0)


def get_span_features_2d(span: Span, **kwargs) -> Floats2d:
    layer_i: int = kwargs.get("layer", -1)
    wp_array = span.doc._.trf_data.arrays[layer_i][span._.trf_row]
    wp_array = span.doc._.trf_get_features_3d(**kwargs)
    row_freqs = numpy.zeros((wp_array.shape[0], 1), dtype="f")
    output = numpy.zeros((len(span), wp_array.shape[1]), dtype="f")
    for token in span:
        for i, j in token._.trf_alignment:
            if i == span._.trf_row:
                output[i] += wp_array[j]
                row_freqs[j] += 1
    output /= row_freqs
    return span.doc._.trf_data.asarray(output)


def get_span_features_3d(span: Span, **kwargs) -> Floats3d:
    trf_data = span.doc._.trf_data
    arrays = []
    layer_indices = kwargs.get("layers", range(len(trf_data.arrays)))
    for i in layer_indices:
        arrays.append(span._.trf_get_features_2d(layer=i))
    output = trf_data.ops.xp.concatenate(arrays)
    return output


def get_token_features_1d(token: Token, **kwargs) -> Floats1d:
    rows = token._.trf_get_features_2d(**kwargs)
    return rows.mean(axis=0)


def get_token_features_2d(token: Token, **kwargs) -> Floats2d:
    layer_i = kwargs.get("layer", -1)
    trf_data = token.doc._.trf_data
    wp_array = trf_data.arrays[layer_i]
    output = wp_array[token._.trf_alignment]
    return output


def get_token_features_3d(token: Token, **kwargs) -> Floats3d:
    trf_data = token.doc._.trf_data
    layer_indices = kwargs.get("layers", list(range(len(trf_data.arrays))))
    arrays = [token._.trf_get_features_2d(layer=i) for i in layer_indices]
    return trf_data.ops.xp.concatenate(arrays)
