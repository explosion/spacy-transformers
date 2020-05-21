from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
import catalogue
import spacy.util
from thinc.api import get_current_ops, CupyOps


class registry(spacy.util.registry):
    span_getters = catalogue.create("spacy", "span_getters", entry_points=True)
    annotation_setters = catalogue.create(
        "spacy", "annotation_setters", entry_points=True
    )


def huggingface_from_pretrained(source, config):
    tokenizer = AutoTokenizer.from_pretrained(source, **config)
    transformer = AutoModel.from_pretrained(source)
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return tokenizer, transformer


def huggingface_tokenize(tokenizer, texts) -> BatchEncoding:
    token_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_masks=True,
        return_lengths=True,
        return_offsets_mapping=False,
        return_tensors="pt",
        return_token_type_ids=None,  # Sets to model default
        pad_to_max_length=True,
    )
    token_data["input_texts"] = [
        tokenizer.convert_ids_to_tokens(list(ids)) for ids in token_data["input_ids"]
    ]
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


def transpose_list(nested_list):
    output = []
    for i, entry in enumerate(nested_list):
        while len(output) < len(entry):
            output.append([None] * len(nested_list))
        for j, x in enumerate(entry):
            output[j][i] = x
    return output
