#!/usr/bin/env python
# coding: utf8


'''


    Example of training spaCy's named entity recognizer, starting off with an
    existing model or a blank model, modified to use the new:

        spacy-pytorch-transformers

    Basically the same code can be found on: 

        https://github.com/explosion/spaCy/blob/master/examples/training/train_ner.py

    Some modifications were made to add the new pipes related to the new models,
    and a different new TRAIN_DATA dataset.

    Compatible with: spaCy v2.0.0+
    Last tested with: v2.1.8

'''


from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data
TRAIN_DATA = [
    ("Uber blew through $1 million a week", {"entities": [(0, 4, 'ORG')]}),
    ("Android Pay expands to Canada", {"entities": [(0, 11, 'PRODUCT'), (23, 30, 'GPE')]}),
    ("Spotify steps up Asia expansion", {"entities": [(0, 8, "ORG"), (17, 21, "LOC")]}),
    ("Google Maps launches location sharing", {"entities": [(0, 11, "PRODUCT")]}),
    ("Google rebrands its business apps", {"entities": [(0, 6, "ORG")]}),
    ("look what i found on google! 😂", {"entities": [(21, 27, "PRODUCT")]}),
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                losses=losses,
            )
        print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# example: 
#main(model='en_pytt_bertbaseuncased_lg',output_dir=None, n_iter=100)

if __name__ == "__main__":
    plac.call(main)

