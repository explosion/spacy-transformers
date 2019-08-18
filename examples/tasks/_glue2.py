import plac
import random
import torch
import spacy
import spacy.util
import tqdm
import numpy
import wasabi
from spacy.gold import GoldParse

from collections import Counter
from pathlib import Path
from spacy.util import minibatch

from spacy_pytorch_transformers._train import train_while_improving
from spacy_pytorch_transformers.hyper_params import get_hyper_params

from glue_util import read_train_data, read_dev_data
from glue_util import describe_task
from metrics import compute_metrics

HP = None
MSG = None


def create_model(model_name, *, task_type, task_name, labels):
    nlp = spacy.load(model_name)
    textcat = nlp.create_pipe("pytt_textcat", config={"architecture": "softmax_class_vector"})
    for label in labels:
        textcat.add_label(label)
    nlp.add_pipe(textcat)
    optimizer = nlp.resume_training()
    return nlp, optimizer


def evaluate(nlp, task, docs_golds, *, eval_batch_size=32):
    tok2vec = nlp.get_pipe("pytt_tok2vec")
    textcat = nlp.get_pipe("pytt_textcat")
    right = 0
    total = 0
    guesses = []
    truths = []
    labels = textcat.labels
    for batch in minibatch(docs_golds, size=eval_batch_size):
        docs, golds = zip(*batch)
        docs = list(textcat.pipe(tok2vec.pipe(docs)))
        for doc, gold in zip(docs, golds):
            guess, _ = max(doc.cats.items(), key=lambda it: it[1])
            truth, _ = max(gold.cats.items(), key=lambda it: it[1])
            if guess not in labels:
                msg = (
                    f"Unexpected label {guess} predicted. "
                    f"Expectded labels: {', '.join(labels)}"
                )
                raise ValueError(msg)
            if truth not in labels:
                msg = (
                    f"Unexpected label {truth} predicted. "
                    f"Expectded labels: {', '.join(labels)}"
                )
                raise ValueError(msg)
            guesses.append(labels.index(guess))
            truths.append(labels.index(truth))
            right += guess == truth
            total += 1
            free_tensors(doc)
    main_name, metrics = compute_metrics(
        task, numpy.array(guesses), numpy.array(truths)
    )
    metrics["_accuracy"] = right / total
    metrics["_right"] = right
    metrics["_total"] = total
    metrics["_main"] = metrics[main_name]
    return metrics[main_name], metrics


def process_data(nlp, task, examples):
    """Set-up Doc and GoldParse objects from the examples. This makes it easier
    to set up text-pair tasks, and also easy to handle datasets with non-real
    tokenization."""
    wordpiecer = nlp.get_pipe("pytt_wordpiecer")
    textcat = nlp.get_pipe("pytt_textcat")
    docs = []
    golds = []
    for eg in examples:
        if eg.text_b:
            assert "\n" not in eg.text_a
            assert "\n" not in eg.text_b
            doc = nlp.make_doc(eg.text_a + "\n" + eg.text_b)
            doc._.pytt_separator = "\n"
        else:
            doc = nlp.make_doc(eg.text_a)
        doc = wordpiecer(doc)
        cats = {label: 0.0 for label in textcat.labels}
        cats[eg.label] = 1.0
        gold = GoldParse(doc, cats=cats)
        docs.append(doc)
        golds.append(gold)
    return list(zip(docs, golds))


def free_tensors(doc):
    doc.tensor = None
    doc._.pytt_last_hidden_state = None
    doc._.pytt_pooler_output = None
    doc._.pytt_all_hidden_states = []
    doc._.pytt_all_attentions = []
    doc._.pytt_d_last_hidden_state = None
    doc._.pytt_d_pooler_output = None
    doc._.pytt_d_all_hidden_states = []
    doc._.pytt_d_all_attentions = []


def save_checkpoint(nlp, output_dir, score, step):
    if score < 1:
        score *= 100
    subdir = output_dir = output_dir / f"{score:.2}_{step}"
    nlp.to_disk(subdir)
    return subdir


def set_seed(seed):
    spacy.util.fix_random_seed(seed)
    torch.manual_seed(seed)


def set_gpu(dont_gpu):
    global MSG
    is_using_gpu = False
    if not dont_gpu:
        is_using_gpu = spacy.prefer_gpu()
        if is_using_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return is_using_gpu


def main(
    model_name: ("Model package", "positional", None, str),
    task: ("Task name", "positional", None, str),
    data_dir: ("Path to data directory", "positional", None, Path),
    output_dir: ("Path to output directory", "positional", None, Path),
    hyper_params: ("Path to hyper params file", "positional", None, Path),
    dont_gpu: ("Dont use GPU, even if available", "flag", "G") = False,
):
    global HP, MSG
    if output_dir.parts[-1] != task:
        output_dir = output_dir / task
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    HP = get_hyper_params(hyper_params)
    MSG = wasabi.Printer()
    set_seed(HP.seed)
    HP.is_using_gpu = set_gpu(dont_gpu)
    MSG.info(f"Use gpu? {HP.is_using_gpu}")

    with MSG.loading("Loading model"):
        nlp, optimizer = create_model(model_name, **describe_task(task))
    
    with MSG.loading("Reading corpus"):
        raw_train_data = read_train_data(data_dir, task)
        raw_dev_data = read_dev_data(data_dir, task)
    with MSG.loading("Tokenizing corpus"):
        train_data = process_data(nlp, task, raw_train_data)
        dev_data = process_data(nlp, task, raw_dev_data)

    # Set up printing
    table_widths = [2, 4, 4]
    MSG.info(f"Training. Initial learn rate: {optimizer.alpha}")
    MSG.row(["#", "Loss", "Score"], widths=table_widths)
    MSG.row(["-" * width for width in table_widths])

    # This sets up a generator, so it doesn't proceed until we loop over it.
    training_step_iterator = train_while_improving(
        nlp,
        train_data,
        lambda: evaluate(nlp, task, dev_data),
        learning_rate=HP.learning_rate,
        batch_size=HP.batch_size,
        weight_decay=HP.weight_decay,
        classifier_lr=HP.classifier_lr,
        dropout=HP.dropout,
        lr_range=HP.lr_range,
        lr_period=HP.lr_period,
        steps_per_batch=HP.steps_per_batch,
        patience=HP.patience,
        eval_every=HP.eval_every
    )
    losses = Counter()
    for batch, info, is_best_checkpoint in training_step_iterator:
        losses.update(info["loss"])
        if is_best_checkpoint is not None:
            loss = info["loss"]["pytt_textcat"]
            score = info["score"]
            step = info["step"]
            # Save checkpoint if this is our current best
            if is_best_checkpoint:
                nlp.to_disk(output_dir)
            # Display stats
            MSG.row([info["step"], f"{loss:.2f}", score], widths=table_widths)
    # Okay, training finished! Print a result-summary table.
    results = info["checkpoints"]
    table_widths = [2, 4, 6]
    MSG.info(f"Best scoring checkpoints")
    MSG.row(["Epoch", "Step", "Score"], widths=table_widths)
    MSG.row(["-" * width for width in table_widths])
    for score, step, epoch in sorted(results, reverse=True)[:10]:
        MSG.row([epoch, step, "%.2f" % (score * 100)], widths=table_widths)


if __name__ == "__main__":
    plac.call(main)
