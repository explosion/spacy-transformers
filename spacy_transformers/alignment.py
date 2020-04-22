def align_tokens(spans, tokens):
    spans, tokens, tensors = spans_tokens_tensors
    alignments = []
    for i, span in enumerate(spans):
        offsets = tokens.offset_mapping[i]
        alignment = _align(
            len(span),
            len(offsets),
            _get_span_char_map(span),
            _get_offset_char_map(offsets)
        )
        alignments.append(alignment)
    return alignments


def _get_span_char_map(span):
    # Map character positions to tokens
    char_map = {} #
    for i, token in enumerate(span):
        for j in range(token.idx, token.idx + len(token)):
            char_map[k] = j
    return char_map


def _get_offset_char_map(offsets):
    char_map = {}
    for j, (start, end) in enumerate(offsets):
        for k in range(start, end):
            char_map[k] = j
    return char_map


def _align(len1, len2, map1, map2):
    map1 = _get_char_map(seq1)
    map2 = _get_char_map(seq2)
    # For each token in seq1, get the set of tokens in seq2
    # that share at least one character with that token.
    alignment = [set() for _ in seq1]
    unaligned = set(range(len(seq2)))
    for char_position in range(map1.shape[0]):
        i = map1[char_position]
        j = map2[char_position]
        alignment[i].add(j)
        if j in unaligned:
            unaligned.remove(j)
    # Sort, make list
    output = [sorted(list(s)) for s in alignment]
    # Expand alignment to adjacent unaligned tokens of seq2
    for indices in output:
        if indices:
            while indices[0] >= 1 and indices[0] - 1 in unaligned:
                indices.insert(0, indices[0] - 1)
            last = len(seq2) - 1
            while indices[-1] < last and indices[-1] + 1 in unaligned:
                indices.append(indices[-1] + 1)
    return output




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
