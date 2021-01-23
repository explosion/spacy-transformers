from typing import List, Dict, Union
from pathlib import Path
import random
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import catalogue
from spacy.util import registry
from thinc.api import get_current_ops, CupyOps
import torch.cuda


# fmt: off
registry.span_getters = catalogue.create("spacy", "span_getters", entry_points=True)
registry.annotation_setters = catalogue.create("spacy", "annotation_setters", entry_points=True)
# fmt: on


def huggingface_from_pretrained(source: Union[Path, str], config: Dict):
    """Create a Huggingface transformer model from pretrained weights. Will
    download the model if it is not already downloaded.

    source (Union[str, Path]): The name of the model or a path to it, such as
        'bert-base-cased'.
    config (dict): Settings to pass to the tokenizer.
    """
    if hasattr(source, "absolute"):
        str_path = str(source.absolute())
    else:
        str_path = source
    tokenizer = AutoTokenizer.from_pretrained(str_path, **config)
    transformer = AutoModel.from_pretrained(str_path)
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return tokenizer, transformer


def huggingface_tokenize(tokenizer, texts: List[str]) -> BatchEncoding:
    """Apply a Huggingface tokenizer to a batch of texts."""
    token_data = tokenizer(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=isinstance(tokenizer, PreTrainedTokenizerFast),
        return_tensors="pt",
        return_token_type_ids=None,  # Sets to model default
        padding="longest",
    )
    token_data["input_texts"] = []
    for i in range(len(token_data["input_ids"])):
        wp_texts = tokenizer.convert_ids_to_tokens(token_data["input_ids"][i])
        token_data["input_texts"].append(wp_texts)
    token_data["pad_token"] = tokenizer.pad_token
    return token_data


def maybe_flush_pytorch_cache(chance: float = 1.0):
    """Flip a coin and decide whether to flush PyTorch's cache. This allows the
    cache to be flushed periodically without maintaining a counter.

    I'm not sure why this is necessary, it shouldn't be. But it definitely does
    help...
    """
    if random.random() < chance and torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_last_hidden(tensors) -> int:
    """Find the index of the hidden layer in a list of activation tensors.
    Internals.
    """
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


def batch_by_length(seqs, max_words: int) -> List[List[int]]:
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order.

    Batches may be at most max_words in size, defined as max sequence length * size.
    """
    # Use negative index so we can get sort by position ascending.
    lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort()
    batches: List[List[int]] = []
    batch: List[int] = []
    for length, i in lengths_indices:
        if not batch:
            batch.append(i)
        elif length * (len(batch) + 1) <= max_words:
            batch.append(i)
        else:
            batches.append(batch)
            batch = [i]
    if batch:
        batches.append(batch)
    # Check lengths match
    assert sum(len(b) for b in batches) == len(seqs)
    # Check no duplicates
    seen = set()
    for b in batches:
        seen.update(id(item) for item in b)
    assert len(seen) == len(seqs)
    batches = [list(sorted(batch)) for batch in batches]
    batches.reverse()
    return batches


def log_gpu_memory(logger, context):
    mem = torch.cuda.memory_allocated() // 1024 ** 2
    logger.info(f"{mem:.1f}: {context}")


def log_batch_size(logger, token_data, is_train):
    batch_size = token_data["input_ids"].shape[0]
    seq_len = token_data["input_ids"].shape[1]
    squared = seq_len ** 2 * batch_size

    if is_train:
        logger.info(f"{batch_size} x {seq_len} ({squared}) update")
    else:
        logger.info(f"{batch_size} x {seq_len} ({squared}) predict")
