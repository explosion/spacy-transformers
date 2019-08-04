import plac
import random
import torch
import spacy
import tqdm

from collections import Counter
from pathlib import Path
from spacy.util import minibatch

from spacy_pytorch_transformers._extra.util import get_hyper_params


def load_data(nlp, task, data_dir):
    train_data = []
    dev_data = []
    return train_data, dev_data


def create_model(model_name, train_data):
    nlp = None
    optimizer = None
    return nlp, optimizer


def train_epoch(nlp, optimizer, train_data):
    global HP
    random.shuffle(train_data)
    batches = minibatch(train_data, size=HP.batch_size)
    for batch in batches:
        texts, annotations = zip(*batch)
        losses = {}
        nlp.update(texts, annotations, sgd=optimizer, drop=HP.dropout, losses=losses)
        yield batch, losses


def evaluate(nlp, task, examples):
    results  = {}
    return results


def print_progress(train_metrics, dev_metrics):
    pass


def maybe_save_checkpoint(args, nlp, results):
    pass


def count_words(data):
    """Make a rough word count, for progress tracking."""
    n = 0
    for text, labels in data:
        n += len(text.split())
    return n


def main(
    model_name: ("Model package", "positional", "model", str),
    task: ("Task name", "positional", None, str),
    data_dir: ("Path to data directory", "positional", None, Path),
    output_dir: ("Path to output directory", "positional", None, Path),
    hyper_params: ("Path to hyper params file", "positional", None, Path) = None,
    dont_gpu: ("Dont use GPU, even if available", "flag", "G") = False,
):
    global HP
    HP = get_hyper_params(hyper_params) 

    spacy.utils.fix_random_seed(HP.seed)
    torch.manual_seed(HP.seed)
    if not dont_gpu:
        is_gpu = spacy.prefer_gpu()
        if is_gpu:
            HP.device = torch.device("cuda")

    train_data, dev_data = load_data(task, data_dir)
    nlp, optimizer = create_model(model_name, train_data)
    
    total_words = count_words(train_data)
    results = []
    for i in range(HP.nr_epoch):
        # Train and evaluate
        losses = Counter()
        with tqdm.tqdm(total=total_words, leave=False) as pbar:
            for batch, loss in train_epoch(nlp, optimizer, train_data):
                pbar.update(count_words(batch))
                losses.update(loss)
        accuracies = evaluate(nlp, dev_data)
        # Track progress and save checkpoints
        print_progress(losses, accuracies)
        results.append((losses, accuracies))
        maybe_save_checkpoint(nlp, results)


if __name__ == "__main__":
    plac.call(main)
