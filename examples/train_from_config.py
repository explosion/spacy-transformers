import plac
from pathlib import Path
from spacy_transformers import install_extensions, TransformerModelByName
from spacy_transformers import Transformer
import spacy_transformers.tok2vec
from spacy.cli.train_from_config import train_from_config


def main(config_path, train_path, eval_path, gpu_id):
    gpu_id = int(gpu_id)
    config_path = Path(config_path)
    train_path = Path(train_path)
    eval_path = Path(eval_path)
    install_extensions()
    train_from_config(config_path, {"train": train_path, "dev": eval_path},
        use_gpu=gpu_id)


if __name__ == "__main__":
    plac.call(main)
