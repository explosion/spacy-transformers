"""Adjust the initialization and serialization of the Transformers
tokenizers, so that they work more nicely with spaCy. Specifically, the
Transformers classes take file paths as arguments to their __init__, which means
we can't easily use to_bytes() and from_bytes() with them.

Additionally, provide a .clean_text() method that is used to match the preprocessing,
so that we can get the alignment working.
"""
from collections import OrderedDict
import spacy
import ftfy
import srsly
import regex
import sentencepiece
from pathlib import Path
import unicodedata
import re

import transformers
from transformers.tokenization_gpt2 import bytes_to_unicode
from transformers.tokenization_bert import BasicTokenizer, WordpieceTokenizer


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
    "added_tokens_decoder",
    "init_kwargs",
    "unique_added_tokens_encoder_list",
]


class SerializationMixin:
    """Provide generic serialization methods, for compatibility with spaCy.
    These expect the tokenizer subclass to provide the following:

    * serialization_fields (List[str]): List of attributes to serialize. All
        attributes should have json-serializable values.
    * finish_deserializing(): A function to be called after from_bytes(),
        to finish setting up the instance.
    """

    def prepare_for_serialization(self):
        self.unique_added_tokens_encoder_list = list(self.unique_added_tokens_encoder)

    def finish_deserializing(self):
        self.unique_added_tokens_encoder = set(self.unique_added_tokens_encoder_list)

    def from_bytes(self, bytes_data, exclude=tuple(), **kwargs):
        msg = srsly.msgpack_loads(bytes_data)
        for field in self.serialization_fields:
            setattr(self, field, msg[field])
        self.finish_deserializing()
        return self

    def to_bytes(self, exclude=tuple(), **kwargs):
        self.prepare_for_serialization()
        msg = OrderedDict()
        for field in self.serialization_fields:
            msg[field] = getattr(self, field, None)
        return srsly.msgpack_dumps(msg)

    def from_disk(self, path, exclude=tuple(), **kwargs):
        with path.open("rb") as file_:
            data = file_.read()
        return self.from_bytes(data, **kwargs)

    def to_disk(self, path, exclude=tuple(), **kwargs):
        data = self.to_bytes(**kwargs)
        with path.open("wb") as file_:
            file_.write(data)


class SerializableBertTokenizer(transformers.BertTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "vocab",
        "do_basic_tokenize",
        "do_lower_case",
        "never_split",
        "tokenize_chinese_chars",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(cls)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.ids_to_tokens = None
        self.basic_tokenizer = None
        self.wordpiece_tokenizer = None
        return self

    def prepare_for_serialization(self):
        if self.basic_tokenizer is not None:
            self.do_lower_case = self.basic_tokenizer.do_lower_case
            self.never_split = self.basic_tokenizer.never_split
            self.tokenize_chinese_chars = self.basic_tokenizer.tokenize_chinese_chars
        super().prepare_for_serialization()

    def finish_deserializing(self):
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=self.do_lower_case,
                never_split=self.never_split,
                tokenize_chinese_chars=self.tokenize_chinese_chars,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token
        )
        super().finish_deserializing()

    def clean_token(self, text):
        if self.do_basic_tokenize:
            text = self.basic_tokenizer._clean_text(text)
        text = text.strip()
        return clean_accents(text)

    def clean_wp_token(self, token):
        return token.replace("##", "", 1).strip()

    def add_special_tokens(self, segments):
        output = []
        for segment in segments:
            output.extend(segment)
            if segment:
                output.append(self.sep_token)
        if output:
            # If we otherwise would have an empty output, don't add cls
            output.insert(0, self.cls_token)
        return output

    def fix_alignment(self, segments):
        """Turn a nested segment alignment into an alignment for the whole input,
        by offsetting and accounting for special tokens."""
        offset = 0
        output = []
        for segment in segments:
            if segment:
                offset += 1
            seen = set()
            for idx_group in segment:
                output.append([idx + offset for idx in idx_group])
                seen.update({idx for idx in idx_group})
            offset += len(seen)
        return output


class SerializableDistilBertTokenizer(
    transformers.DistilBertTokenizer, SerializationMixin
):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "vocab",
        "do_basic_tokenize",
        "do_lower_case",
        "never_split",
        "tokenize_chinese_chars",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(cls)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.ids_to_tokens = None
        self.basic_tokenizer = None
        self.wordpiece_tokenizer = None
        return self

    def prepare_for_serialization(self):
        if self.basic_tokenizer is not None:
            self.do_lower_case = self.basic_tokenizer.do_lower_case
            self.never_split = self.basic_tokenizer.never_split
            self.tokenize_chinese_chars = self.basic_tokenizer.tokenize_chinese_chars
        super().prepare_for_serialization()

    def finish_deserializing(self):
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=self.do_lower_case,
                never_split=self.never_split,
                tokenize_chinese_chars=self.tokenize_chinese_chars,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token
        )
        super().finish_deserializing()

    def clean_token(self, text):
        if self.do_basic_tokenize:
            text = self.basic_tokenizer._clean_text(text)
        text = text.strip()
        return clean_accents(text)

    def clean_wp_token(self, token):
        return token.replace("##", "", 1).strip()

    def add_special_tokens(self, segments):
        output = []
        for segment in segments:
            output.extend(segment)
            if segment:
                output.append(self.sep_token)
        if output:
            # If we otherwise would have an empty output, don't add cls
            output.insert(0, self.cls_token)
        return output

    def fix_alignment(self, segments):
        """Turn a nested segment alignment into an alignment for the whole input,
        by offsetting and accounting for special tokens."""
        offset = 0
        output = []
        for segment in segments:
            if segment:
                offset += 1
            seen = set()
            for idx_group in segment:
                output.append([idx + offset for idx in idx_group])
                seen.update({idx for idx in idx_group})
            offset += len(seen)
        return output


class SerializableGPT2Tokenizer(transformers.GPT2Tokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "encoder",
        "_bpe_ranks",
        "errors",
        "_regex_pattern",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(cls)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.byte_encoder = None
        self.byte_decoder = None
        self.bpe_ranks = {}
        self.cache = None
        self.pat = None
        return self

    def finish_deserializing(self):
        self.bpe_ranks = deserialize_bpe_ranks(self._bpe_ranks)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.pat = regex.compile(self._regex_pattern, flags=regex.V0)
        super().finish_deserializing()

    def prepare_for_serialization(self):
        self._regex_pattern = self.pat.pattern
        self._bpe_ranks = serialize_bpe_ranks(self.bpe_ranks)
        super().prepare_for_serialization()

    def clean_token(self, text):
        text = clean_extended_unicode(text)
        return text.strip()

    def clean_wp_token(self, text):
        text = text.replace("\u0120", "", 1)
        text = text.replace("\u010a", "", 1)
        text = ftfy.fix_text(text)
        text = clean_extended_unicode(text)
        return text.strip()

    def add_special_tokens(self, segments):
        output = []
        for segment in segments:
            if segment:
                output.append(self.bos_token)
            output.extend(segment)
            if segment:
                output.append(self.eos_token)
        return output

    def fix_alignment(self, segments):
        """Turn a nested segment alignment into an alignment for the whole input,
        by offsetting and accounting for special tokens."""
        offset = 0
        output = []
        for segment in segments:
            if segment:
                offset += 1
            seen = set()
            for idx_group in segment:
                output.append([idx + offset for idx in idx_group])
                seen.update({idx for idx in idx_group})
            offset += len(seen)
            if segment:
                offset += 1
        return output


class SerializableXLMTokenizer(transformers.XLMTokenizer, SerializationMixin):
    _replace_re = re.compile(r"[\s\.\-`'\";]+")
    serialization_fields = list(BASE_CLASS_FIELDS) + ["encoder", "_bpe_ranks"]

    @classmethod
    def blank(cls):
        self = cls.__new__(cls)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.nlp = None
        self.fix_text = None
        self.cache = None
        self.decoder = {}
        self.bpe_ranks = {}
        return self

    def finish_deserializing(self):
        self.bpe_ranks = deserialize_bpe_ranks(self._bpe_ranks)
        self.nlp = spacy.blank("en")
        self.fix_text = ftfy.fix_text
        self.cache = {}
        self.decoder = {v: k for k, v in self.encoder.items()}
        super().finish_deserializing()

    def prepare_for_serialization(self):
        self._bpe_ranks = serialize_bpe_ranks(self.bpe_ranks)
        super().prepare_for_serialization()

    def clean_token(self, text):
        # Model seems to just strip out all unicode so we need to do this, too,
        # instead of calling clean_accents etc.
        text = ftfy.fix_text(text)
        text = re.sub(r"&(#\d+|#x[a-f\d]+)", "", text)  # malformed HTML entities
        text = clean_extended_unicode(text)
        text = self._replace_re.sub("", text)
        return text.strip()

    def clean_wp_token(self, text):
        text = ftfy.fix_text(text)
        text = clean_extended_unicode(text)
        text = self._replace_re.sub("", text)
        return text.replace("</w>", "").strip()

    def add_special_tokens(self, segments):
        # See https://github.com/facebookresearch/XLM/issues/113
        output = []
        for segment in segments:
            if segment:
                output.append(self.bos_token)
            output.extend(segment)
            if segment:
                output.append(self.eos_token)
        return output

    def fix_alignment(self, segments):
        """Turn a nested segment alignment into an alignment for the whole input,
        by offsetting and accounting for special tokens."""
        offset = 0
        output = []
        for segment in segments:
            if segment:
                offset += 1
            seen = set()
            for idx_group in segment:
                output.append([idx + offset for idx in idx_group])
                seen.update({idx for idx in idx_group})
            offset += len(seen)
            if segment:
                offset += 1
        return output


class SerializableXLNetTokenizer(transformers.XLNetTokenizer, SerializationMixin):
    _replace_re = re.compile(r"[\s'\";]+")
    _replacements = [("º", "o"), *zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")]
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "do_lower_case",
        "remove_space",
        "keep_accents",
        "vocab_bytes",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(cls)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.sp_model = None
        return self

    def prepare_for_serialization(self):
        if hasattr(self, "vocab_file"):
            vocab_path = Path(self.vocab_file)
            with vocab_path.open("rb") as f:
                self.vocab_bytes = f.read()
        super().prepare_for_serialization()

    def finish_deserializing(self):
        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.LoadFromSerializedProto(self.vocab_bytes)
        super().finish_deserializing()

    def clean_token(self, text):
        text = clean_fractions(text)
        text = clean_accents(text)
        for a, b in self._replacements:
            text = text.replace(a, b)
        text = clean_extended_unicode(text)
        text = self._replace_re.sub("", text)
        return text.strip()

    def clean_wp_token(self, text):
        # Note: The special control character is \u2581
        text = clean_accents(text)
        for a, b in self._replacements:
            text = text.replace(a, b)
        text = clean_extended_unicode(text)
        text = self._replace_re.sub("", text)
        return text.strip()

    def add_special_tokens(self, segments):
        output = []
        for segment in segments:
            output.extend(segment)
            if segment:
                output.append(self.eos_token)
        if output:
            output.append(self.cls_token)
        return output

    def fix_alignment(self, segments):
        """Turn a nested segment alignment into an alignment for the whole input,
        by offsetting and accounting for special tokens."""
        offset = 0
        output = []
        for segment in segments:
            seen = set()
            for idx_group in segment:
                output.append([idx + offset for idx in idx_group])
                seen.update({idx for idx in idx_group})
            offset += len(seen)
            if segment:
                offset += 1
        return output


class SerializableRobertaTokenizer(transformers.RobertaTokenizer, SerializationMixin):
    serialization_fields = list(BASE_CLASS_FIELDS) + [
        "encoder",
        "_bpe_ranks",
        "errors",
        "_regex_pattern",
    ]

    @classmethod
    def blank(cls):
        self = cls.__new__(cls)
        for field in self.serialization_fields:
            setattr(self, field, None)
        self.byte_encoder = None
        self.byte_decoder = None
        self.bpe_ranks = {}
        self.cache = None
        self.pat = None
        return self

    def finish_deserializing(self):
        self.bpe_ranks = deserialize_bpe_ranks(self._bpe_ranks)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.pat = regex.compile(self._regex_pattern, flags=regex.V0)
        super().finish_deserializing()

    def prepare_for_serialization(self):
        self._regex_pattern = self.pat.pattern
        self._bpe_ranks = serialize_bpe_ranks(self.bpe_ranks)
        super().prepare_for_serialization()

    def clean_token(self, text):
        text = clean_extended_unicode(text)
        return text.strip()

    def clean_wp_token(self, text):
        text = text.replace("\u0120", "", 1)
        text = text.replace("\u010a", "", 1)
        text = ftfy.fix_text(text)
        text = clean_extended_unicode(text)
        return text.strip()

    def add_special_tokens(self, segments):
        # A RoBERTa sequence pair has the following format: [CLS] A [SEP][SEP] B [SEP]
        output = []
        for segment in segments:
            if output:
                output.append(self.sep_token)
            output.extend(segment)
            if segment:
                output.append(self.sep_token)
        if output:
            # If we otherwise would have an empty output, don't add cls
            output.insert(0, self.cls_token)
        return output

    def fix_alignment(self, segments):
        """Turn a nested segment alignment into an alignment for the whole input,
        by offsetting and accounting for special tokens."""
        offset = 0
        output = []
        for segment in segments:
            if segment:
                offset += 1
            seen = set()
            for idx_group in segment:
                output.append([idx + offset for idx in idx_group])
                seen.update({idx for idx in idx_group})
            offset += len(seen)
            if segment:
                offset += 1
        return output


def clean_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def clean_fractions(text):
    chars = []
    for c in text:
        try:
            name = unicodedata.name(c)
        except ValueError:
            chars.append(c)
            continue
        name = unicodedata.name(c)
        if name.startswith("VULGAR FRACTION"):
            chars.append(unicodedata.normalize("NFKC", c))
        else:
            chars.append(c)
    return "".join(chars)


def clean_extended_unicode(text):
    return "".join(i for i in text if 31 < ord(i) < 127)


def serialize_bpe_ranks(data):
    return [{"key": list(key), "value": value} for key, value in data.items()]


def deserialize_bpe_ranks(data):
    return {tuple(item["key"]): item["value"] for item in data}
