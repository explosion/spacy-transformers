

def set_tensors(docs: List[Doc], trfout: TransformerOutput) -> None:
    wp_tensor = outputs.tensors[-1]
    for doc in docs:
        # Count how often each word-piece token is represented. This allows
        # a weighted sum, so that we can make sure doc.tensor.sum()
        # equals wp_tensor.sum(). Do this with sensitivity to boundary tokens
        wp_indices = _align_doc(doc, trfout.spans, trfout.tokens)
        doc.tensor = _get_aligned_tensor(wp_indices, wp_tensor)


def get_aligned_tensor(
    wp_indices: List[List[Tuple[int, int]]],
    wp_tensor: Floats3d
) -> Floats2d:
    """Given an alignment array, extract features from another tensor, with rows
    weighted to account for multiple occurrence.
    """
    align_sizes = numpy.zeros((wp_tensor.shape[0], wp_tensor.shape[1]), dtype="f")
    for token_alignment in alignment:
        for i, j in token_alignment:
            align_sizes[i, j] += 1
    xp = get_array_module(outputs.tensors[-1])
    tensor = xp.zeros((len(doc), wp_tensor.shape[-1]), dtype="f")
    wp_weighted = wp_tensor / xp.array(align_sizes, dtype="f").reshape((-1, 1))
    # TODO: Obviously incrementing the rows individually is bad. How
    # to do in one shot without blowing up the memory?
    for i, word_piece_slice in enumerate(wp_indices):
        tensor[i] = wp_weighted[word_piece_slice,].sum(0)
    return tensor


def set_hooks(docs, outputs, alignment):
    for doc in docs:
        doc.user_hooks["vector"] = get_doc_vector_via_tensor
        doc.user_span_hooks["vector"] = get_span_vector_via_tensor
        doc.user_token_hooks["vector"] = get_token_vector_via_tensor
        doc.user_hooks["similarity"] = get_similarity_via_tensor
        doc.user_span_hooks["similarity"] = get_similarity_via_tensor
        doc.user_token_hooks["similarity"] = get_similarity_via_tensor


def get_doc_vector_via_tensor(doc):
    return doc.tensor.sum(axis=0)


def get_span_vector_via_tensor(span):
    return span.doc.tensor[span.start : span.end].sum(axis=0)


def get_token_vector_via_tensor(token):
    return token.doc.tensor[token.i]


def get_similarity_via_tensor(doc1, doc2):
    v1 = doc1.vector
    v2 = doc2.vector
    xp = get_array_module(v1)
    return xp.dot(v1, v2) / (doc1.vector_norm * doc2.vector_norm)
