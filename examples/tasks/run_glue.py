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

from spacy_pytorch_transformers.util import cyclic_triangular_rate, PIPES, ATTRS
from spacy_pytorch_transformers.hyper_params import get_hyper_params

from glue_util import read_train_data, read_dev_data
from glue_util import describe_task
from metrics import compute_metrics


def create_model(model_name, *, task_type, task_name, labels):
    nlp = spacy.load(model_name)
    textcat = nlp.create_pipe(PIPES.textcat, config={"architecture": HP.textcat_arch})
    for label in labels:
        textcat.add_label(label)
    nlp.add_pipe(textcat)
    optimizer = nlp.resume_training()
    optimizer.alpha = HP.learning_rate
    optimizer.max_grad_norm = HP.max_grad_norm
    optimizer.eps = HP.adam_epsilon
    optimizer.L2 = 0.0
    return nlp, optimizer


def train_epoch(nlp, optimizer, train_data):
    # This isn't the recommended code -- but it's the easiest way to do the
    # experiment for now.
    global HP
    random.shuffle(train_data)
    batches = minibatch(train_data, size=HP.batch_size)
    tok2vec = nlp.get_pipe(PIPES.tok2vec)
    textcat = nlp.get_pipe(PIPES.textcat)
    for batch in batches:
        docs, golds = zip(*batch)
        tokvecs, backprop_tok2vec = tok2vec.begin_update(docs, drop=HP.dropout)
        losses = {}
        tok2vec.set_annotations(docs, tokvecs)
        textcat.update(docs, golds, drop=HP.dropout, sgd=optimizer, losses=losses)
        backprop_tok2vec(docs, sgd=optimizer)
        yield batch, losses
        for doc in docs:
            free_tensors(doc)


def evaluate(nlp, task, docs_golds):
    tok2vec = nlp.get_pipe(PIPES.tok2vec)
    textcat = nlp.get_pipe(PIPES.textcat)
    right = 0
    total = 0
    guesses = []
    truths = []
    labels = textcat.labels
    for batch in minibatch(docs_golds, size=HP.eval_batch_size):
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
    wordpiecer = nlp.get_pipe(PIPES.wordpiecer)
    textcat = nlp.get_pipe(PIPES.textcat)
    docs = []
    golds = []
    for eg in examples:
        if eg.text_b:
            assert "\n" not in eg.text_a
            assert "\n" not in eg.text_b
            doc = nlp.make_doc(eg.text_a + "\n" + eg.text_b)
            doc._.set(ATTRS.separator, "\n")
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
    doc._.set(ATTRS.last_hidden_state, None)
    doc._.set(ATTRS.pooler_output, None)
    doc._.set(ATTRS.all_hidden_states, [])
    doc._.set(ATTRS.all_attentions, [])
    doc._.set(ATTRS.d_last_hidden_state, None)
    doc._.set(ATTRS.d_pooler_output, None)
    doc._.set(ATTRS.d_all_hidden_states, [])
    doc._.set(ATTRS.d_all_attentions, [])


def main(
    model_name: ("Model package", "positional", None, str),
    task: ("Task name", "positional", None, str),
    data_dir: ("Path to data directory", "positional", None, Path),
    output_dir: ("Path to output directory", "positional", None, Path),
    hyper_params: ("Path to hyper params file", "positional", None, Path) = None,
    dont_gpu: ("Dont use GPU, even if available", "flag", "G") = False,
):
    global HP
    HP = get_hyper_params(hyper_params)
    msg = wasabi.Printer()

    spacy.util.fix_random_seed(HP.seed)
    torch.manual_seed(HP.seed)
    if not dont_gpu:
        is_using_gpu = spacy.prefer_gpu()
        msg.info(f"Use gpu? {is_using_gpu}")
        if is_using_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

    with msg.loading("Reading corpus"):
        raw_train_data = read_train_data(data_dir, task)
        raw_dev_data = read_dev_data(data_dir, task)
    with msg.loading("Loading model"):
        nlp, optimizer = create_model(model_name, **describe_task(task))
    with msg.loading("Tokenizing corpus"):
        train_data = process_data(nlp, task, raw_train_data)
        dev_data = process_data(nlp, task, raw_dev_data)

    nr_batch = len(train_data) // HP.batch_size
    # Set up printing
    table_widths = [2, 4, 4]
    msg.info(f"Training. Initial learn rate: {optimizer.alpha}")
    msg.row(["#", "Loss", "Score"], widths=table_widths)
    msg.row(["-" * width for width in table_widths])
    # Set up learning rate schedule
    learn_rates = cyclic_triangular_rate(
        HP.learning_rate / HP.lr_range,
        HP.learning_rate * HP.lr_range,
        nr_batch * HP.lr_period,
    )
    optimizer.trf_lr = next(learn_rates)
    optimizer.trf_weight_decay = HP.weight_decay
    optimizer.trf_use_swa = HP.use_swa
    # This sets the learning rate for the Thinc layers, i.e. just the final
    # softmax. By keeping this LR high, we avoid a problem where the model
    # spends too long flat, which harms the transfer learning.
    optimizer.alpha = 0.001
    step = 0
    if HP.eval_every < 1:
        HP.eval_every = nr_batch
    pbar = tqdm.tqdm(total=HP.eval_every, leave=False)
    results = []
    epoch = 0
    while True:
        # Train and evaluate
        losses = Counter()
        for batch, loss in train_epoch(nlp, optimizer, train_data):
            pbar.update(1)
            losses.update(loss)

            optimizer.trf_lr = next(learn_rates)
            if step and (step % HP.eval_every) == 0:
                with nlp.use_params(optimizer.averages):
                    main_score, accuracies = evaluate(nlp, task, dev_data)
                results.append((main_score, step, epoch))
                msg.row(
                    [str(step), "%.2f" % losses[PIPES.textcat], main_score],
                    widths=table_widths,
                )
                pbar.close()
                pbar = tqdm.tqdm(total=HP.eval_every, leave=False)
            step += 1
        epoch += 1
        # Stop if no improvement in HP.patience checkpoints
        if results:
            best_score, best_step, best_epoch = max(results)
            if ((step - best_step) // HP.eval_every) >= HP.patience:
                break

    table_widths = [2, 4, 6]
    msg.info(f"Best scoring checkpoints")
    msg.row(["Epoch", "Step", "Score"], widths=table_widths)
    msg.row(["-" * width for width in table_widths])
    for score, step, epoch in sorted(results, reverse=True)[:10]:
        msg.row([epoch, step, "%.2f" % (score * 100)], widths=table_widths)


if __name__ == "__main__":
    plac.call(main)
