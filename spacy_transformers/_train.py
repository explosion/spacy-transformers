import random
from spacy.util import minibatch
from .util import cyclic_triangular_rate


def train_while_improving(
    nlp,
    train_data,
    evaluate,
    *,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    classifier_lr: float,
    dropout: float,
    lr_range: int,
    lr_period: int,
    steps_per_batch: int,
    patience: int,
    eval_every: int
):
    """Train until an evaluation stops improving. Works as a generator,
    with each iteration yielding a tuple `(batch, info, is_best_checkpoint)`,
    where info is a dict, and is_best_checkpoint is in [True, False, None] --
    None indicating that the iteration was not evaluated as a checkpoint.
    The evaluation is conducted by calling the evaluate callback, which should

    Positional arguments:
        nlp: The spaCy pipeline to evaluate.
        train_data (Sized[Tuple[Union[unicode, Doc], Union[GoldParse, dict]]]):
            The training data.
        evaluate (Callable[[], Tuple[float, Any]]): A callback to perform evaluation.
            The callback should take no arguments and return a tuple
            `(main_score, other_scores)`. The main_score should be a float where
            higher is better. other_scores can be any object.

    The following hyper-parameters (passed as keyword arguments) may need to be
    adjusted for your problem:

        learning_rate (float): Central learning rate to cycle around. 1e-5 is
            often good, but it can depend on the batch size.
        batch_size (int): The number of examples per iteration. Try 32 and 128.
            Larger batch size makes the gradient estimation more accurate, but
            means fewer updates to the model are made per pass over the data.
            With more passes over the data, the updates are less noisy, which
            can lead to inferior generalisation and longer training times.
            The batch size is expressed in number of examples, so the best
            value will depend on the number of words per example in your data.
            If your documents are long, you can use a lower batch size (because
            you're using more information to estimate the gradients). With a
            large dataset and short examples, try a batch size of 128, possibly
            setting a higher value for `steps_per_batch` if you run out of memory
            (see below). Also try a smaller batch size like 32.

    The following hyper-parameters can affect accuracy, but generalize fairly
    well and probably don't need to be tuned:

        weight_decay (float): The weight decay for the AdamW optimizer. 0.005
            is a good value.
        classifier_lr (float): The learning rate for the classifier parameters,
            which must be trained from scatch on each new problem. A value of
            0.001 is good -- it's best for the classifier to learn much faster
            than the rest of the network, which is initialised from the language
            model.
        lr_range (int): The range to vary the learning rate over during training.
            The learning rate will cycle between learning_rate / lr_range and
            learning_rate * lr_range. 2 is good.
        lr_period (int): How many epochs per min-to-max period in the cycle.
            2 is good. Definitely don't set patience < lr_period * 2 --- you
            want at least one full cycle before you give up.

    The following hyper-parameters impact compute budgets.

        patience (int): How many evaluations to allow without improvement
            before giving up. e.g. if patience is 5 and the best evaluation
            was the tenth one, the loop may halt from evaluation 15 onward.
            The loop only halts after a full epoch though, so depending on the
            evaluation frequency, more than `patience` checkpoints may occur
            after the best one. 10 is good.
        eval_every (int): How often to evaluate, in number of iterations.
            max(100, (len(train_data) // batch_size) // 10) is good -- i.e.
            either 100, or about every 10% of a full epoch. For small training
        steps_per_batch (int): Accumulate gradients over a number of steps for
            each batch. This allows you to use a higher batch size with less memory,
            at the expense of potentially slower compute costs. If you don't need
            it, just set it to 1.

    Every iteration, the function yields out a tuple with:

    * batch: A zipped sequence of Tuple[Doc, GoldParse] pairs.
    * info: A dict with various information about the last update (see below).
    * is_best_checkpoint: A value in None, False, True, indicating whether this
        was the best evaluation so far. You should use this to save the model
        checkpoints during training. If None, evaluation was not conducted on
        that iteration. False means evaluation was conducted, but a previous
        evaluation was better.

    The info dict provides the following information:

        epoch (int): How many passes over the data have been completed.
        step (int): How many steps have been completed.
        score (float): The main score form the last evaluation.
        other_scores: : The other scores from the last evaluation.
        loss: The accumulated losses throughout training.
        checkpoints: A list of previous results, where each result is a (score, step, epoch) tuple.
    """
    nr_eg = len(train_data)
    nr_batch = nr_eg // batch_size
    steps_per_epoch = nr_batch * steps_per_batch
    optimizer = nlp.resume_training()
    learn_rates = cyclic_triangular_rate(
        learning_rate / lr_range, learning_rate * lr_range, steps_per_epoch
    )
    optimizer.trf_lr = next(learn_rates)
    optimizer.trf_weight_decay = HP.weight_decay
    # This sets the learning rate for the Thinc layers, i.e. just the final
    # softmax. By keeping this LR high, we avoid a problem where the model
    # spends too long flat, which harms the transfer learning.
    optimizer.alpha = HP.classifier_lr
    epoch = 0
    step = 0
    results = []
    while True:
        random.shuffle(train_data)
        for batch in minibatch(train_data, size=(batch_size // steps_per_batch)):
            optimizer.trf_lr = next(learn_rates)
            docs, golds = zip(*batch)
            losses = {}
            nlp.update(
                docs,
                golds,
                drop=HP.dropout,
                losses=losses,
                sgd=(optimizer if (step % steps_per_batch == 0) else None),
            )
            if step != 0 and not (step % (eval_every * steps_per_batch)):
                with nlp.use_params(optimizer.averages):
                    score, other_scores = evaluate()
                results.append((score, step, epoch))
                is_best_checkpoint = score == max(results)[0]
            else:
                score, other_scores = (None, None)
                is_best_checkpoint = None
            info = {
                "epoch": epoch,
                "step": step,
                "score": score,
                "other_scores": other_scores,
                "loss": losses,
                "checkpoints": results,
            }
            yield batch, info, is_best_checkpoint
            step += 1
        epoch += 1
        # Stop if no improvement in HP.patience checkpoints
        if results:
            best_score, best_step, best_epoch = max(results)
            if ((step - best_step) // HP.eval_every) >= HP.patience:
                break
