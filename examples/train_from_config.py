import plac
from pathlib import Path
from thinc.api import use_pytorch_for_gpu_memory
from spacy.cli.train_from_config import train_from_config
import spacy.util


def main(config_path, train_path, eval_path, gpu_id):
    gpu_id = int(gpu_id)
    config_path = Path(config_path)
    train_path = Path(train_path)
    eval_path = Path(eval_path)
    if gpu_id >= 0:
        spacy.util.use_gpu(gpu_id)
        use_pytorch_for_gpu_memory()
    train_from_config(config_path, {"train": train_path, "dev": eval_path})


if __name__ == "__main__":
    plac.call(main)
