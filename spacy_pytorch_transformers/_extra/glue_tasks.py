# coding: utf8
"""
Utilities to work with the GLUE shared task data.

Adapted from Huggingface's pytorch-transformers.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple


@dataclass
class InputExample:
    """A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: int
    text_a: str
    text_b: str = ""
    label: str = ""


def read_train_data(data_dir: Path, task: str) -> List[InputExample]:
    return PROCESSORS[task]().get_train_examples(data_dir)


def read_dev_data(data_dir: Path, task: str) -> List[InputExample]:
    return PROCESSORS[task]().get_dev_examples(data_dir)


def describe_task(task: str) -> dict:
    T = PROCESSORS[task]()
    return dict(task_name=T.name, task_type=T.task, labels=T.labels)


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    name: str
    task: str
    labels: Tuple[str, ...]
    train_filename: str
    dev_filename: str
    subdir: str

    @property
    def train_name(self) -> str:
        """The partition-name used for the training data. Usually matches
        train_filename but without the extension.
        """
        return self.train_filename.rsplit(".", 1)[0]

    @property
    def dev_name(self) -> str:
        """The partition-name used for the dev data. Usually matches
        dev_filename but without the extension.
        """
        return self.dev_filename.rsplit(".", 1)[0]

    def get_train_examples(self, data_dir: Path) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the train set."""
        filename = data_dir / self.subdir / self.train_filename
        return list(self._read_examples(filename, self.train_name))

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        filename = data_dir / self.subdir / self.dev_filename
        return list(self._read_examples(filename, self.dev_name))

    def _read_examples(
        self, path: Path, set_type: str, quote=None
    ) -> Iterator[InputExample]:
        """Creates examples for the training and dev sets."""
        with path.open("r", encoding="utf-8-sig") as file_:
            reader = csv.reader(file_, delimiter="\t", quotechar=quote)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                yield self.create_example(i, set_type, line)

    def create_example(self, i: int, set_type: str, line: List[str]) -> InputExample:
        raise NotImplementedError


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    name = "mrpc"
    subdir = "MRPC"
    task = "classification"

    labels = ("0", "1")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{i}-{set_type}"
        return InputExample(guid, line[3], line[4], line[0])


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    name = "mnli"
    subdir = "MNLI"
    task = "classification"
    labels = ("contradiction", "entailment", "neutral")
    train_filename = "train.tsv"
    dev_filename = "dev_matched.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{set_type}-{line[0]}"
        return InputExample(guid, line[8], line[9], line[-1])


class MnliMismatchedProcessor(DataProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    name = "mnli-mm"
    subdir = "MNLI"
    task = "classification"

    labels = ("contradiction", "entailment", "neutral")
    train_filename = "train.tsv"
    dev_filename = "dev_matched.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{set_type}-{line[0]}"
        return InputExample(guid, line[8], line[9], line[-1])


class ColaProcessor(DataProcessor):
    name = "cola"
    subdir = "CoLA"
    task = "classification"

    labels = ("0", "1")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{set_type}-{i}"
        return InputExample(guid, line[3], "", line[1])


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    name = "sst2"
    subdir = "SST-2"
    task = "classification"

    labels = ("0", "1")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{set_type}-{i}"
        return InputExample(guid, line[0], "", line[1])


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    name = "sts-b"
    subdir = "STS-B"
    task = "regression"

    labels = ("",)
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{set_type}-{line[0]}"
        return InputExample(guid, line[7], line[8], line[-1])


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    name = "qqp"
    subdir = "QQP"
    task = "classification"

    labels = ("0", "1")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        guid = f"{set_type}-{line[0]}"
        try:
            text_a = line[3]
            text_b = line[4]
            label = line[5]
        except IndexError:
            return None
        return InputExample(guid, text_a, text_b, label)


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    name = "qnli"
    subdir = "QNLI"
    task = "classification"

    labels = ("entailment", "not_entailment")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    @property
    def dev_name(self):
        return "dev_matched"

    def create_example(self, i, set_type, line):
        """Creates examples for the training and dev sets."""
        guid = f"{set_type}-{line[0]}"
        return InputExample(guid, line[1], line[2], line[-1])


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    name = "rte"
    subdir = "RTE"
    task = "classification"
    labels = ("entailment", "not_entailment")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        """Creates examples for the training and dev sets."""
        guid = f"{set_type}-{line[0]}"
        return InputExample(guid, line[1], line[2], line[-1])


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    name = "wnli"
    subdir = "WNLI"
    task = "classification"
    labels = ("0", "1")
    train_filename = "train.tsv"
    dev_filename = "dev.tsv"

    def create_example(self, i, set_type, line):
        guid = (f"{set_type}-{line[0]}",)
        return InputExample(guid, line[1], line[2], line[-1])


PROCESSORS_LIST = [
    ColaProcessor,
    MnliProcessor,
    MnliMismatchedProcessor,
    MrpcProcessor,
    Sst2Processor,
    StsbProcessor,
    QqpProcessor,
    QnliProcessor,
    RteProcessor,
    WnliProcessor,
]

PROCESSORS = {proc.name: proc for proc in PROCESSORS_LIST}
