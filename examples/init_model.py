#!/usr/bin/env python
import plac
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer
from spacy_pytorch_transformers import PyTT_TokenVectorEncoder


@plac.annotations(
    path=("Output path", "positional", None, str),
    name=("Name of pre-trained model", "option", "n", str),
    lang=("Language code to use", "option", "l", str),
)
def main(path, name="bert-base-uncased", lang="en"):
    nlp = PyTT_Language(pytt_name=name, meta={"lang": lang})
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, name))
    nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, name))
    nlp.to_disk(path)
    print(f"Saved '{name}' ({lang}) model to {path}")


if __name__ == "__main__":
    plac.call(main)
