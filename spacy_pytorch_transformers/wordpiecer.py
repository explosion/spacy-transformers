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

    def require_model(self):
        return None

    def update(self, *args, **kwargs):
        return None

    #def to_bytes(self, exclude=tuple(), **kwargs):
    #    msg = {
    #        "cfg": self.cfg,
    #        "tokenizer": self.pytt_tokenizer.to_bytes()
    #    }
    #    return srsly.msgpack_dumps(msg)
#
#    def from_bytes(self, bytes_data, exclude=tuple(), **kwargs):
#        msg = srsly.msgpack_loads(byte_data)
#        self.cfg = msg["cfg"]
##        PyTT_Tokenizer = get_pytt_tokenizer(self.cfg["pytt_name"])
#        self.pytt_tokenizer = PyTT_Tokenizer.blank().from_bytes(msg["tokenizer"])
#        return self

#    def to_disk(self, path, exclude=tuple(), **kwargs):
#        if not path.exists():
#            path.mkdir()
#        srsly.write_json(path / "cfg", self.cfg)
#        srsly.write_msgpack(path / "tokenizer.msg", self.pytt_tokenizer.to_bytes())

#    def from_disk(self, path, exclude=tuple(), **kwargs):
#        self.cfg = srsly.read_json(path / "cfg")
#        msg = srsly.read_msgpack(path / "tokenizer.msg")
#        PyTT_Tokenizer = get_pytt_tokenizer(self.cfg["pytt_name"])
#        self.pytt_tokenizer = PyTT_Tokenizer.blank().from_bytes(msg)
#        return self
