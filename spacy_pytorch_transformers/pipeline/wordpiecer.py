from spacy.pipeline import Pipe
from spacy.util import minibatch
import re
import numpy

from ..util import get_pytt_tokenizer, flatten_list, is_special_token


class PyTT_WordPiecer(Pipe):
    """spaCy pipeline component to assign PyTorch-Transformers word-piece
    tokenization to the Doc, which can then be used by the token vector
    encoder. Note that this component doesn't modify spaCy's tokenization. It
    only sets extension attributes and aligns the tokens."""

    name = "pytt_wordpiecer"

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def from_pretrained(cls, vocab, pytt_name, **cfg):
        model = get_pytt_tokenizer(pytt_name).from_pretrained(pytt_name)
        return cls(vocab, model=model, pytt_name=pytt_name, **cfg)

    @classmethod
    def Model(cls, pytt_name, **kwargs):
        return get_pytt_tokenizer(pytt_name).blank()

    def __init__(self, vocab, model=True, **cfg):
        """Initialize the component.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
        model: Not used here.
        **cfg: Optional config parameters.
        """
        self.vocab = vocab
        self.cfg = cfg
        self.model = model

    def __call__(self, doc):
        """Apply the pipe to one document. The document is
        modified in-place, and returned.

        Both __call__ and pipe should delegate to the `predict()`
        and `set_annotations()` methods.
        """
        self.require_model()
        scores = self.predict([doc])
        self.set_annotations([doc], scores)
        return doc

    @property
    def alignment_strategy(self):
        """Mostly used for debugging to make the WordPiecer raise error on
        misaligned tokens and prevent forcing alignment.
        """
        retry_alignment = self.cfg.get("retry_alignment", True)
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
        for doc in docs:
            output.append([])
            for sent in doc.sents:
                tokens = self.model.tokenize(sent.text)
                if tokens:
                    output[-1].append(self.model.add_special_tokens(tokens))
                else:
                    output[-1].append(tokens)
        return output

    def set_annotations(self, docs, outputs, tensors=None):
        """Assign the extracted tokens and IDs to the Doc objects.

        docs (iterable): A batch of `Doc` objects.
        outputs (iterable): A batch of outputs.
        """
        # Set model.max_len to some high value, to avoid annoying prints.
        max_len = self.model.max_len
        self.model.max_len = 1e12
        retry, force = self.alignment_strategy
        for doc, output in zip(docs, outputs):
            offset = 0
            doc_word_pieces = []
            doc_alignment = []
            doc_word_piece_ids = []
            for sent, wp_tokens in zip(doc.sents, output):
                spacy_tokens = [self.model.clean_token(w.text) for w in sent]
                new_wp_tokens = [self.model.clean_wp_token(t) for t in wp_tokens]
                assert len(wp_tokens) == len(new_wp_tokens)
                sent_align = align_word_pieces(spacy_tokens, new_wp_tokens, retry=retry)
                if sent_align is None:
                    if not force:
                        spacy_string = "".join(spacy_tokens).lower()
                        wp_string = "".join(new_wp_tokens).lower()
                        print("spaCy:", spacy_string)
                        print("WP:", wp_string)
                        raise AssertionError((spacy_string, wp_string))
                    # As a final fallback, we resort to word-piece tokenizing
                    # the spaCy tokens individually, to make the alignment
                    # trivial.
                    wp_tokens, sent_align = _tokenize_individual_tokens(
                        self.model, sent
                    )
                # We need to align into the flattened document list, instead
                # of just into this sentence. So offset by number of wp tokens.
                for token_align in sent_align:
                    for i in range(len(token_align)):
                        token_align[i] += offset
                offset += len(wp_tokens)
                doc_alignment.extend(sent_align)
                doc_word_pieces.extend(wp_tokens)
                doc_word_piece_ids.extend(self.model.convert_tokens_to_ids(wp_tokens))
            assert len(doc_alignment) == len(doc)
            max_aligned = max(flatten_list(doc_alignment), default=0)
            assert max_aligned <= len(doc_word_pieces)
            doc._.pytt_word_pieces = doc_word_piece_ids
            doc._.pytt_word_pieces_ = doc_word_pieces
            doc._.pytt_alignment = doc_alignment
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
    offset = 0
    while wp_tokens and is_special_token(wp_tokens[0]):
        wp_tokens.pop(0)
        offset += 1
    while wp_tokens and is_special_token(wp_tokens[-1]):
        wp_tokens.pop(-1)
    if not wp_tokens:
        return [[] for _ in spacy_tokens]
    elif not spacy_tokens:
        return []
    # Check alignment
    spacy_string = "".join(spacy_tokens).lower()
    wp_string = "".join(wp_tokens).lower()
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
                return _align(spacy_tokens, wp_tokens, offset)
        else:
            print("spaCy:", spacy_string)
            print("WP:", wp_string)
            raise AssertionError((spacy_string, wp_string))
        # If either we're not trying the fallback alignment, or the fallback
        # fails, we return None. This tells the wordpiecer to align by
        # calling the sub-tokenizer on the spaCy tokens.
        return None
    output = _align(spacy_tokens, wp_tokens, offset)
    return output


def _align(seq1, seq2, offset):
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
    # Add offset
    for indices in output:
        for i in range(len(indices)):
            indices[i] += offset
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
    # Figure out whether we're adding special tokens
    if model.add_special_tokens(["the"])[0] != "the":
        offset += 1
    for token in sent:
        subtokens = model.tokenize(token.text)
        wp_tokens.extend(subtokens)
        sent_align.append([offset + i for i in range(len(subtokens))])
        offset += len(subtokens)
    return model.add_special_tokens(wp_tokens), sent_align
