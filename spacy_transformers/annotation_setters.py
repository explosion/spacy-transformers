def set_tensors(docs, outputs, alignment):
    for i, doc in enumerate(docs):
        wp_tensor = outputs.tensors[-1][i]
        # Count how often each word-piece token is represented. This allows
        # a weighted sum, so that we can make sure doc.tensor.sum()
        # equals wp_tensor.sum(). Do this with sensitivity to boundary tokens
        wp_rows, align_sizes = alignment[i]

        xp = get_array_module(wp_tensor)
        doc.tensor = xp.zeros((len(doc), outputs.width), dtype="f")
        wp_weighted = wp_tensor / xp.array(align_sizes, dtype="f").reshape((-1, 1))
        # TODO: Obviously incrementing the rows individually is bad. How
        # to do in one shot without blowing up the memory?
        for i, word_piece_slice in enumerate(wp_rows):
            doc.tensor[i] = wp_weighted[word_piece_slice,].sum(0)


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
