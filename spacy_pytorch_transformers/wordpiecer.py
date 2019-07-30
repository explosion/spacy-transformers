from spacy.pipeline import Pipe
from spacy.util import minibatch

from .util import get_pytt_tokenizer, align_word_pieces, flatten_list


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
    def from_pretrained(cls, vocab, pytt_name):
        model = get_pytt_tokenizer(pytt_name).from_pretrained(pytt_name)
        return cls(vocab, model=model, pytt_name=pytt_name)

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
                output[-1].append(self.model.add_special_tokens(tokens))
        return output

    def set_annotations(self, docs, outputs, tensors=None):
        """Assign the extracted tokens and IDs to the Doc objects.

        docs (iterable): A batch of `Doc` objects.
        outputs (iterable): A batch of outputs.
        """
        for doc, output in zip(docs, outputs):
            offset = 0
            doc_word_pieces = []
            doc_alignment = []
            doc_word_piece_ids = []
            for sent, wp_tokens in zip(doc.sents, output):
                spacy_tokens = [self.model.clean_token(w.text) for w in sent]
                new_wp_tokens = self.model.clean_wp_tokens(wp_tokens)
                assert len(wp_tokens) == len(new_wp_tokens)
                sent_align = align_word_pieces(spacy_tokens, new_wp_tokens)
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
            max_aligned = max(flatten_list(doc_alignment))
            assert max_aligned <= len(doc_word_pieces)
            doc._.pytt_word_pieces = doc_word_piece_ids
            doc._.pytt_word_pieces_ = doc_word_pieces
            doc._.pytt_alignment = doc_alignment
