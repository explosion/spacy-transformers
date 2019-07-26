<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy wrapper for Huggingface's PyTorch Transformers

This package provides [spaCy](https://spacy.io) model pipelines that wrap [Huggingface's `pytorch-transformers`](https://github.com/huggingface/pytorch-transformers)
package, so you can use them in spaCy. The result is convenient access to
state-of-the-art transformer architectures, such as BERT, GPT2, XLNet, etc.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/11/master.svg?logo=azure-devops&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=11)
[![PyPi](https://img.shields.io/pypi/v/spacy-pytorch-transformers.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-pytorch-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-pytorch-transformers/all.svg?style=flat-square)](https://github.com/explosion/spacy-pytorch-transformers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## Quickstart

The following will download and install the weights, PyTorch, and other
required dependencies:

```bash
python -m spacy download en_transformer_bertbaseuncased_pytorch
```

Once all that's downloaded (over 1gb), you can load it as a normal pipeline,
and access the outputs directly via extension attributes.

```python
import spacy
nlp = spacy.load("en_transformer_bertbaseuncased_pytorch")
doc = nlp("Hello this is some text")
doc._.word_piece_tokens
doc._.word_piece_ids
doc._.transformer_outputs
doc._.transformer_cls_vector
```

## Transfer learning

The main use-case for pretrained transformer models is transfer learning. You
load in the pretrained weights, and start training on your data. This package
has custom pipeline components that make this especially easy.

```python
# This stuff will all be set by you.
TRAIN_DATA = [
    ("text1", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}})
]
DEV_DATA = [
    ("text2", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}})
]
EXC_CLS = True
NB_EPOCH = 10 # Number of training epochs


import spacy
from spacy.util import minibatch

nlp = spacy.load("en_transformer_bert-base-uncased_pytorch")
textcat = nlp.create_pipe("textcat", config={"exclusive_classes": EXC_CLS})
for label in LABELS:
    textcat.add_label(label)

nlp.add_pipe(textcat)
optimizer = nlp.resume_training()

for i in range(NB_EPOCH):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for batch in minibatch(TRAIN_DATA, size=BATCH_SIZE):
        texts, cats = zip(*batch)
        nlp.update(texts, cats, optimizer=optimizer, losses=losses)
    scores = nlp.evaluate(dev_texts, dev_cats)
    print_progress(i, scores, losses)

nlp.to_disk(OUTPUT_DIR)
```
