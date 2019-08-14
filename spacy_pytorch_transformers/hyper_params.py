from configparser import ConfigParser


FIELDS = dict(
    max_seq_length=int,
    dropout=float,
    batch_size=int,
    eval_batch_size=int,
    learning_rate=float,
    lr_range=float,
    lr_period=int,
    weight_decay=float,
    adam_epsilon=float,
    max_grad_norm=float,
    num_train_epochs=int,
    max_steps=int,
    warmup_steps=int,
    seed=int,
    textcat_arch=str,
    eval_every=int,
    use_learn_rate_schedule=int,
    use_swa=int,
    patience=int,
)


class HyperParams:
    pass


def get_hyper_params(path):
    cfg = ConfigParser()
    with path.open("r", encoding="utf8") as file_:
        cfg.read_string(file_.read())
    values = cfg["defaults"]
    HP = HyperParams()
    for field, type_ in FIELDS.items():
        setattr(HP, field, type_(values[field]))
    return HP
