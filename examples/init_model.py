#!/usr/bin/env python
import plac
from wasabi import Printer
from spacy_pytorch_transformers import TransformersLanguage, TransformersWordPiecer
from spacy_pytorch_transformers import TransformersTok2Vec


@plac.annotations(
    path=("Output path", "positional", None, str),
    name=("Name of pre-trained model", "option", "n", str),
    lang=("Language code to use", "option", "l", str),
)
def main(path, name="bert-base-uncased", lang="en"):
    msg = Printer()
    msg.info(f"Creating model for '{name}' ({lang})")
    with msg.loading(f"Setting up the pipeline..."):
        nlp = TransformersLanguage(pytt_name=name, meta={"lang": lang})
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        nlp.add_pipe(TransformersWordPiecer.from_pretrained(nlp.vocab, name))
        nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, name))
    msg.good("Initialized the model pipeline")
    nlp.to_disk(path)
    msg.good(f"Saved '{name}' ({lang})")
    msg.text(f"Pipeline: {nlp.pipe_names}")
    msg.text(f"Location: {path}")
    with msg.loading("Verifying model loads..."):
        nlp.from_disk(path)
    msg.good("Model loads!")


if __name__ == "__main__":
    plac.call(main)
