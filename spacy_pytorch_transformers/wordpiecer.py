from spacy.pipeline import Pipe

from .util import get_pytt_tokenizer, align_word_pieces


class PyTT_WordPiecer(Pipe):
    """spaCy pipeline component to assign PyTorch-Transformers word-piece
    tokenization to the Doc, which can then be used by the token vector
    encoder. Note that this component doesn't modify spaCy's tokenization. It
    only sets extension attributes and aligns the tokens."""

    name = "pytt_wordpiecer"

    def __init__(self, vocab, model=True, **cfg):
        """Initialize the component.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
        model: Not used here.
        **cfg: Optional config parameters.
        """
        name = cfg.get("pytt_name")
        if not name:
            raise ValueError("Need pytt_name argument, e.g. 'bert-base-uncased'")
        pytt_cls = get_pytt_tokenizer(name)
        self.model = pytt_cls.from_pretrained(name)

    def predict(self, docs):
        """Run the word-piece tokenizer on a batch of docs and return the
        extracted strings.

        docs (iterable): A batch of Docs to process.
        RETURNS (tuple): A (strings, None) tuple.
        """
        bos = self.model.cls_token
        sep = self.model.sep_token
        strings = []
        for doc in docs:
            strings.append([bos] + self.model.tokenize(doc.text) + [sep])
        return strings, None

    def set_annotations(self, docs, outputs, tensors=None):
        """Assign the extracted tokens and IDs to the Doc objects.

        docs (iterable): A batch of `Doc` objects.
        outputs (iterable): A batch of outputs.
        """
        for doc, output in zip(docs, outputs):
            doc._.pytt_word_pieces_ = output
            doc._.pytt_word_pieces = self.model.convert_tokens_to_ids(output)
            doc._.pytt_alignment = align_word_pieces([w.text for w in doc], output)
