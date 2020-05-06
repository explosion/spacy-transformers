from typing import List
from spacy.tokens import Doc

from .types import TransformerData, FullTransformerBatch, BatchEncoding


def install_extensions():
    Doc.set_extension("trf_data", default=TransformerData.empty())


def get_doc_spans(docs):
    return [doc[:] for doc in docs]

def configure_strided_spans(window, stride):
    def get_strided_spans(docs):
        spans = []
        for doc in docs:
            start = 0
            for i in range(len(doc) // stride):
                spans.append(doc[start : start + window])
                start += stride
            if start < len(doc):
                spans.append(doc[start : ])
        return spans
    return get_strided_spans

def get_sent_spans(docs):
    sents = []
    for doc in docs:
        sents.extend(doc.sents)
    return sents


def huggingface_tokenize(tokenizer, texts) -> BatchEncoding:
    token_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=False,
        return_tensors="pt",
        return_token_type_ids=None,  # Sets to model default
    )
    # Work around https://github.com/huggingface/transformers/issues/3224
    extra_token_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=True,
        return_tensors=None,
        return_token_type_ids=None,  # Sets to model default
    )
    # There seems to be some bug where it's flattening single-entry batches?
    if len(texts) == 1:
        token_data["offset_mapping"] = [extra_token_data["offset_mapping"]]
    else:
        token_data["offset_mapping"] = extra_token_data["offset_mapping"]
    return token_data


def slice_hf_tokens(inputs: BatchEncoding, start: int, end: int) -> BatchEncoding:
    output = {}
    for key, value in inputs.items():
        if not hasattr(value, "__getitem__"):
            output[key] = value
        else:
            output[key] = value[start:end]
    return output


def find_last_hidden(tensors) -> int:
    for i, tensor in reversed(list(enumerate(tensors))):
        if len(tensor.shape) == 3:
            return i
    else:
        raise ValueError("No 3d tensors")


def null_annotation_setter(docs: List[Doc], trf_data: FullTransformerBatch) -> None:
    """Set no additional annotations on the Doc objects."""
    pass
