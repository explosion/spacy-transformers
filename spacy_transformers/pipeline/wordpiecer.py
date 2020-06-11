from spacy.pipeline import Pipe
from spacy.util import minibatch
from tokenizations import get_alignments
import re
import numpy

from ..util import get_tokenizer, flatten_list, get_sents, PIPES, ATTRS


class TransformersWordPiecer(Pipe):
    """spaCy pipeline component to assign transformer word-piece
    tokenization to the Doc, which can then be used by the token vector
    encoder. Note that this component doesn't modify spaCy's tokenization. It
    only sets extension attributes and aligns the tokens."""

    name = PIPES.wordpiecer
    factory = PIPES.wordpiecer

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def from_pretrained(cls, vocab, trf_name, **cfg):
        model = get_tokenizer(trf_name).from_pretrained(trf_name)
        return cls(vocab, model=model, trf_name=trf_name, **cfg)

    @classmethod
    def Model(cls, trf_name, **kwargs):
        return get_tokenizer(trf_name).blank()

    def __init__(self, vocab, model=True, **cfg):
        """Initialize the component.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
        model: Not used here.
        **cfg: Optional config parameters.
        """
        self.vocab = vocab
        self.cfg = cfg
        self.model = model
        self._num_added_tokens = None

    def __call__(self, doc):
        """Apply the pipe to one document. The document is
        modified in-place, and returned.

        Both __call__ and pipe should delegate to the `predict()`
        and `set_annotations()` methods.
        """
        self.require_model()
        pieces = self.predict([doc])
        self.set_annotations([doc], pieces)
        return doc

    @property
    def alignment_strategy(self):
        """Mostly used for debugging to make the WordPiecer raise error on
        misaligned tokens and prevent forcing alignment.
        """
        retry_alignment = self.cfg.get("retry_alignment", False)
        force_alignment = self.cfg.get("force_alignment", True)
        return (retry_alignment, force_alignment)

    def pipe(self, stream, batch_size=128):
        """Process Doc objects as a stream and assign the extracted features.

        stream (iterable): A stream of Doc objects.
        batch_size (int): The number of texts to buffer.
        YIELDS (spacy.tokens.Doc): Processed Docs in order.
        """
        for docs in minibatch(stream, size=batch_size):
            docs = list(docs)
            outputs = self.predict(docs)
            self.set_annotations(docs, outputs)
            for doc in docs:
                yield doc

    def predict(self, docs):
        """Run the word-piece tokenizer on a batch of docs and return the
        extracted strings.

        docs (iterable): A batch of Docs to process.
        RETURNS (tuple): A (strings, None) tuple.
        """
        output = []
        max_seq_length = self.model.max_len - self.num_added_tokens()
        for doc in docs:
            doc_words = []
            doc_align = []
            offset = 0
            for sent in get_sents(doc):
                sent_words = []
                sent_align = []
                for segment in sent._.get(ATTRS.segments):
                    seg_words = self.model.tokenize(segment.text)
                    seg_words, seg_align = self._align(
                        segment, seg_words, offset=offset
                    )
                    # max_seq_length per sentence is calculated relative to the
                    # index of the first aligned segment within the sentence
                    align_start_id = 0
                    for align in seg_align:
                        if len(align) >= 1:
                            align_start_id = align[0]
                            break
                    seg_words = seg_words[:max_seq_length]
                    for i, align in enumerate(seg_align):
                        if (
                            len(align) >= 1
                            and align[-1] < align_start_id + max_seq_length
                        ):
                            continue
                        seg_align[i] = [
                            x for x in align if x < align_start_id + max_seq_length
                        ]
                    assert len(segment) == len(seg_align)
                    sent_words.append(seg_words)
                    sent_align.append(seg_align)
                sw_flat = self.model.add_special_tokens(sent_words)
                sa_flat = self.model.fix_alignment(sent_align)
                doc_words.extend(sw_flat)
                doc_align.extend(sa_flat)
                offset += len(sw_flat)
            output.append((doc_words, doc_align))
        return output

    def _align(self, segment, wp_tokens, *, offset=0):
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

    def num_added_tokens(self):
        # GPT2 returns 0 for `tokenizer.num_added_tokens()` but
        # `tokenizer.add_special_tokens()` adds two '<|endoftext|>' tokens,
        # so determine this once initially in practice
        if self._num_added_tokens is None:
            words = ["a", "b", "c"]
            words_with_added = self.model.add_special_tokens([words])
            self._num_added_tokens = len(words_with_added) - len(words)
        return self._num_added_tokens

    def set_annotations(self, docs, outputs, tensors=None):
        """Assign the extracted tokens and IDs to the Doc objects.

        docs (iterable): A batch of `Doc` objects.
        outputs (iterable): A batch of outputs.
        """
        # Set model.max_len to some high value, to avoid annoying prints.
        max_len = self.model.max_len
        self.model.max_len = 1e12
        for doc, (wordpieces, alignment) in zip(docs, outputs):
            doc._.set(ATTRS.word_pieces_, wordpieces)
            doc._.set(ATTRS.word_pieces, self.model.convert_tokens_to_ids(wordpieces))
            doc._.set(ATTRS.alignment, alignment)
            nr_word = len(doc._.get(ATTRS.word_pieces))
            words_per_sent = sum(
                len(sent._.get(ATTRS.word_pieces)) for sent in get_sents(doc)
            )
            if nr_word != words_per_sent:
                print([repr(w.text) for w in doc])
                for sent in get_sents(doc):
                    print(sent._.get(ATTRS.word_pieces_))
                    for w in sent:
                        print(w.text, w._.get(ATTRS.alignment))
                print(doc._.get(ATTRS.word_pieces_))
                self.model.max_len = max_len
                raise ValueError(
                    f"Error calculating word pieces for sentences. Total number "
                    f"of wordpieces in the doc was {nr_word}, but adding up the "
                    f"wordpieces for its sentences we get {words_per_sent}. This "
                    f"means there's a bug in the extension attributes or "
                    f"the tokenizer.add_special_tokens() logic, often when "
                    f"a spaCy sentence aligns against 0 wordpieces."
                )
        self.model.max_len = max_len

    def use_params(self, params):
        yield


alpha_re = re.compile(r"[^A-Za-z]+")


def align_word_pieces(spacy_tokens, wp_tokens, retry=True):
    """Align tokens against word-piece tokens. The alignment is returned as a
    list of lists. If alignment[3] == [4, 5, 6], that means that spacy_tokens[3]
    aligns against 3 tokens: wp_tokens[4], wp_tokens[5] and wp_tokens[6].
    All spaCy tokens must align against at least one element of wp_tokens.
    """
    spacy_tokens = list(spacy_tokens)
    wp_tokens = list(wp_tokens)
    if not wp_tokens:
        return [[] for _ in spacy_tokens]
    elif not spacy_tokens:
        return []
    # Check alignment
    spacy_string = "".join(spacy_tokens).lower()
    wp_string = "".join(wp_tokens).lower()
    if not spacy_string and not wp_string:
        return None
    if spacy_string != wp_string:
        if retry:
            # Flag to control whether to apply a fallback strategy when we
            # don't align, of making more aggressive replacements. It's not
            # clear whether this will lead to better or worse results than the
            # ultimate fallback strategy, of calling the sub-tokenizer on the
            # spaCy tokens. Probably trying harder to get alignment is good:
            # the ultimate fallback actually *changes what wordpieces we
            # return*, so we get (potentially) different results out of the
            # transformer. The more aggressive alignment can only change how we
            # map those transformer features to tokens.
            spacy_tokens = [alpha_re.sub("", t) for t in spacy_tokens]
            wp_tokens = [alpha_re.sub("", t) for t in wp_tokens]
            spacy_string = "".join(spacy_tokens).lower()
            wp_string = "".join(wp_tokens).lower()
            if spacy_string == wp_string:
                return _align(spacy_tokens, wp_tokens)
        # If either we're not trying the fallback alignment, or the fallback
        # fails, we return None. This tells the wordpiecer to align by
        # calling the sub-tokenizer on the spaCy tokens.
        return None
    output = _align(spacy_tokens, wp_tokens)
    if len(set(flatten_list(output))) != len(wp_tokens):
        return None
    return output


def _align(seq1, seq2):
    # Map character positions to tokens
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


def _get_char_map(seq):
    char_map = numpy.zeros((sum(len(token) for token in seq),), dtype="i")
    offset = 0
    for i, token in enumerate(seq):
        for j in range(len(token)):
            char_map[offset + j] = i
        offset += len(token)
    return char_map


def _tokenize_individual_tokens(model, sent):
    # As a last-chance strategy, run the wordpiece tokenizer on the
    # individual tokens, so that alignment is trivial.
    wp_tokens = []
    sent_align = []
    offset = 0
    for token in sent:
        if token.text.strip():
            subtokens = model.tokenize(token.text)
            wp_tokens.extend(subtokens)
            sent_align.append([i + offset for i in range(len(subtokens))])
            offset += len(subtokens)
        else:
            sent_align.append([])
    return wp_tokens, sent_align
