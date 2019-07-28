"""Adjust the initialization and serialization of the PyTorch Transformers 
tokenizers, so that they work more nicely with spaCy. Specifically, the PyTT
classes take file paths as arguments to their __init__, which means we can't
easily use to_bytes() and from_bytes() with them.
"""
from collections import OrderedDict
import spacy
import ftfy

import pytorch_transformers as pytt
from pytorch_transformers.tokenization_gpt2 import bytes_to_unicode


BASE_CLASS_FIELDS = [
    "_bos_token",
    "_eos_token",
    "_unk_token",
    "_sep_token",
    "_pad_token",
    "_cls_token",
    "_mask_token",
    "_additional_special_tokens",
    "max_len",
    "added_tokens_encoder",
    "added_tokens_decoder"
]


class SerializationMixin:
    """Provide generic serialization methods, for compatibility with spaCy.
    These expect the tokenizer subclass to provide the following:

    * serialization_fields (List[str]): List of attributes to serialize. All
        attributes should have json-serializable values.
    * finish_deserializing(): A function to be called after from_bytes(),
        to finish setting up the instance.
    """
    def from_bytes(self, bytes_data, exclude=tuple(), **kwargs):
        msg = srsly.msgpack_loads(bytes_data)
        for field in self.serialization_fields:
            setattr(self, field, msg[field])
        self.finish_deserializing()

    def to_bytes(self, exclude=tuple(), **kwargs):
        msg = OrderedDict()
        for field in self.serialization_fields:
            msg[field] = getattr(self, field)
        return srsly.msgpack_dumps(msg)

    def from_disk(self, path, exclude=tuple(), **kwargs):
        with (path / "pytt_tokenizer.msg").open("rb") as file_:
            data = file_.read()
        return self.from_bytes(data, **kwargs)

    def to_disk(self, path, exclude=tuple(), **kwargs):
        data = self.to_bytes(data, **kwargs)
        with (path / "pytt_tokenizer.msg").open("wb") as file_:
            file_.write(data)


class SerializableBertTokenizer(pytt.BertTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "vocab",
        "do_basic_tokenize",
        "do_lower_case",
        "never_split",
        "tokenize_chinese_chars"
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(self)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.ids_to_tokens = None
        self.basic_tokenizer = None
        self.wordpiece_tokenizer = None

    def finish_deserializing(self):
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=self.do_lower_case,
                never_split=self.never_split,
                tokenize_chinese_chars=self.tokenize_chinese_chars
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
            unk_token=self.unk_token)



class SerializableGPT2Tokenizer(pytt.GPT2Tokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "encoder",
        "bpe_ranks",
        "errors",
        "_regex_pattern",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(self)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.byte_encoder = None
        self.byte_decoder = None
        self.cache = None
        self.pat = None

    def finish_deserializing(self):
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.pat = re.compile(self._regex_pattern)


class SerializableOpenAIGPTTokenizer(pytt.OpenAIGPTTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "encoder",
        "bpe_ranks",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(self)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.nlp = None
        self.fix_text = None
        self.cache = None
        self.decoder = {}
 
    def finish_deserializing(self):
        self.nlp = spacy.blank("en")
        self.fix_text = ftfy.fix_text
        self.cache = {}
        self.decoder = {v:k for k,v in self.encoder.items()}
 

class SerializableTransfoXLTokenizer(pytt.TransfoXLTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "counter",
        "special",
        "min_freq",
        "max_size",
        "lower_case",
        "delimiter",
        "never_split",
        "idx2sym",
        "eos_idx"
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(self)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.sym2idx = {}

 
    def finish_deserializing(self):
        self.sym2idx = {sym: i for i, sym in enumerate(idx2sym)} 


class SerializableXLMTokenizer(pytt.XLMTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "encoder",
        "bpe_ranks",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(self)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.nlp = None
        self.fix_text = None
        self.cache = None
        self.decoder = {}
 
    def finish_deserializing(self):
        self.nlp = spacy.blank("en")
        self.fix_text = ftfy.fix_text
        self.cache = {}
        self.decoder = {v:k for k,v in self.encoder.items()}
 

class SerializableXLNetTokenizer(pytt.XLNetTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "do_lower_case",
        "remove_space",
        "keep_accents",
        "vocab_bytes"
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(self)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.sp_model = None

    def finish_deserializing(self):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.LoadFromSerializedProto(self.vocab_bytes)


