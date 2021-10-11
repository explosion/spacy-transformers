import pytest
from spacy_transformers.util import huggingface_from_pretrained
from spacy_transformers.util import huggingface_tokenize


def test_deprecation_warnings():
    with pytest.warns(DeprecationWarning):
        tokenizer, transformer = huggingface_from_pretrained(
            "distilbert-base-uncased", {}
        )
    with pytest.warns(DeprecationWarning):
        token_data = huggingface_tokenize(tokenizer, ["a", "b", "c"])
