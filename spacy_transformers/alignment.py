from tokenizations import get_alignments


def align_forward(model, spans_tokens_tensors, is_train):
    spans, tokens, tensors = spans_tokens_tensors
    alignment = []
    # TODO: Offset?
    for i, span in enumerate(spans):
        alignment.append(align(span, tokens, i))
    return alignment


def align(segment, wp_tokens, *, offset=0):
    spacy_tokens = [w.text for w in segment]
    a2b, b2a = get_alignments(spacy_tokens, wp_tokens)
    # a2b must contain the boundary of `segment` (head and last token index)
    # so insert them when they are missed.
    if a2b and b2a:
        if len(b2a[0]) == 0:
            a2b[0].insert(0, 0)
        if len(b2a[-1]) == 0:
            a2b[-1].append(len(b2a) - 1)
    a2b = [[i + offset for i in a] for a in a2b]
    return wp_tokens, a2b
