from spacy.pipeline import Pipe
from spacy.util import to_bytes, from_bytes, to_disk, from_disk
import srsly

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

    def __init__(self, vocab, model=True, **cfg):
        """Initialize the component.

        vocab (spacy.vocab.Vocab): The spaCy vocab to use.
        model: Not used here.
        **cfg: Optional config parameters.
        """
        self.vocab = vocab
        self.cfg = cfg
        self.model = model
        self.pytt_tokenizer = None
        if cfg.get("pytt_name"):
            self.load_tokenizer()

    def load_tokenizer(self):
        name = self.cfg.get("pytt_name")
        if not name:
            raise ValueError("Need pytt_name argument, e.g. 'bert-base-uncased'")
        pytt_cls = get_pytt_tokenizer(name)
        self.pytt_tokenizer = pytt_cls.from_pretrained(name)

    def predict(self, docs):
        """Run the word-piece tokenizer on a batch of docs and return the
        extracted strings.

        docs (iterable): A batch of Docs to process.
        RETURNS (tuple): A (strings, None) tuple.
        """
        bos = self.pytt_tokenizer.cls_token
        sep = self.pytt_tokenizer.sep_token
        strings = []
        for doc in docs:
            strings.append([bos] + self.pytt_tokenizer.tokenize(doc.text) + [sep])
        return strings, None

    def set_annotations(self, docs, outputs, tensors=None):
        """Assign the extracted tokens and IDs to the Doc objects.

        docs (iterable): A batch of `Doc` objects.
        outputs (iterable): A batch of outputs.
        """
        for doc, output in zip(docs, outputs):
            doc._.pytt_word_pieces_ = output
            doc._.pytt_word_pieces = self.pytt_tokenizer.convert_tokens_to_ids(output)
            doc._.pytt_alignment = align_word_pieces([w.text for w in doc], output)

    def require_model(self):
        return None

    def update(self, *args, **kwargs):
        return None

    def to_bytes(self, exclude=tuple(), **kwargs):
        serialize = {"cfg": lambda: srsly.json_dumps(self.cfg)}
        return to_bytes(serialize, exclude)

    def from_bytes(self, bytes_data, exclude=tuple(), **kwargs):
        deserialize = {"cfg": lambda b: self.cfg.update(srsly.json_loads(b))}
        from_bytes(bytes_data, deserialize, exclude)
        self.load_tokenizer()

    def to_disk(self, path, exclude=tuple(), **kwargs):
        serialize = {"cfg": lambda p: srsly.write_json(p, self.cfg)}
        return to_disk(path, serialize, exclude)

    def from_disk(self, path, exclude=tuple(), **kwargs):
        _load_cfg = lambda p: srsly.read_json(p) if p.exists() else {}
        deserialize = {"cfg": lambda p: self.cfg.update(_load_cfg(p))}
        from_disk(path, deserialize, exclude)
        self.load_tokenizer()
