from spacy.pipeline import Pipe

from .util import get_pytt_tokenizer, align_word_pieces


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

    def predict(self, docs):
        """Run the word-piece tokenizer on a batch of docs and return the
        extracted strings.

        docs (iterable): A batch of Docs to process.
        RETURNS (tuple): A (strings, None) tuple.
        """
        bos = self.model.cls_token
        sep = self.model.sep_token
        output = []
        for doc in docs:
            output.append([])
            for sent in doc.sents:
                output[-1].append([bos] + self.model.tokenize(sent.text) + [sep])
        return output, None

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
                spacy_tokens = [w.text for w in sent]
                sent_align = align_word_pieces(spacy_tokens, wp_tokens)
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
            max_aligned = max(max(token_align) for token_align in doc_alignment)
            assert max_aligned <= len(doc_word_pieces)
            doc._.pytt_word_pieces = doc_word_piece_ids
            doc._.pytt_word_pieces_ = doc_word_pieces
            doc._.pytt_alignment = doc_alignment
