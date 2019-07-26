<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy wrapper for PyTorch Transformers

This package provides [spaCy](https://spacy.io) model pipelines that wrap [Huggingface's `pytorch-transformers`](https://github.com/huggingface/pytorch-transformers)
package, so you can use them in spaCy. The result is convenient access to
state-of-the-art transformer architectures, such as BERT, GPT2, XLNet, etc.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/11/master.svg?logo=azure-devops&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=11)
[![PyPi](https://img.shields.io/pypi/v/spacy-pytorch-transformers.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-pytorch-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-pytorch-transformers/all.svg?style=flat-square)](https://github.com/explosion/spacy-pytorch-transformers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## 🚀 Quickstart

The following will download and install the weights, PyTorch, and other
required dependencies:

```bash
python -m spacy download en_transformer_bertbaseuncased_pytorch
```

Once all that's downloaded (over 1gb), you can load it as a normal pipeline,
and access the outputs directly via extension attributes.

## 📖 Usage

### Extension attributes

This wrapper sets the following [custom extension attributes](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Doc`, `Span` and `Token` objects:

| Name                   | Type | Description                                                                                                                    |
| ---------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------ |
| `._.pytt_alignment`    |      |                                                                                                                                |
| `._.pytt_word_pieces`  |      | A Torch tensor of word-piece IDs.                                                                                              |
| `._.pytt_word_pieces_` |      | The string forms of the word-piece IDs.                                                                                        |
| `._.pytt_outputs`      |      | All outputs produced by the PyTorch Transformer model.                                                                         |
| `._.pytt_gradients`    |      | Gradients of the pytt_outputs. These get incremented during `nlp.update`, and then cleared at the end once the update is made. |

The values can be accessed via the `._` attribute. For example:

```python
doc = nlp("This is a text.")
print(doc._.pytt_word_pieces)
```

## 🎛 API

### <kbd>class</kbd> `PyTT_TokenVectorEncoder`

spaCy pipeline component to use PyTorch-Transformers models. The component assigns the output of the transformer to the `doc._.pytt_outputs` extension attribute. We also calculate an alignment between the word-piece
tokens and the spaCy tokenization, so that we can use the last hidden states
to set the `doc.tensor` attribute. When multiple word-piece tokens align to
the same spaCy token, the spaCy token receives the sum of their values.

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.from_pretrained`

| Name        | Type                      | Description |
| ----------- | ------------------------- | ----------- |
| `name`      |                           |             |
| `**cfg`     | -                         |             |
| **RETURNS** | `PyTT_TokenVectorEncoder` |             |

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.Model`

| Name        | Type                 | Description |
| ----------- | -------------------- | ----------- |
| `name`      |                      |             |
| `**cfg`     | -                    |             |
| **RETURNS** | `thinc.neural.Model` |             |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.__init__`

| Name        | Type                          | Description |
| ----------- | ----------------------------- | ----------- |
| `name`      |                               |             |
| `model`     | `thinc.neural.Model` / `True` |             |
| `**cfg`     | -                             |             |
| **RETURNS** | `PyTT_TokenVectorEncoder`     |             |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.__call__`

| Name        | Type               | Description |
| ----------- | ------------------ | ----------- |
| `doc`       | `spacy.tokens.Doc` |             |
| **RETURNS** | `spacy.tokens.Doc` |             |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.pipe`

| Name         | Type               | Description                                            |
| ------------ | ------------------ | ------------------------------------------------------ |
| `stream`     | iterable           | A stream of documents.                                 |
| `batch_size` | int                | The number of texts to buffer. Defaults to `128`.      |
| **YIELDS**   | `spacy.tokens.Doc` | Processed documents in the order of the original text. |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.begin_update`

| Name        | Type     | Description           |
| ----------- | -------- | --------------------- |
| `docs`      | iterable | A batch of documents. |
| `drop`      | float    | The dropout rate.     |
| `**cfg`     | -        |                       |
| **RETURNS** |          |                       |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.predict`

| Name        | Type     | Description           |
| ----------- | -------- | --------------------- |
| `docs`      | iterable | A batch of documents. |
| **RETURNS** |          |                       |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.set_annotations`

| Name      | Type     | Description           |
| --------- | -------- | --------------------- |
| `docs`    | iterable | A batch of documents. |
| `outputs` |          |                       |

### <kbd>class</kbd> `PyTT_Language`

A subclass of [`spacy.Language`](https://spacy.io/api/language) that holds a
PyTorch-Transformer (PyTT) pipeline. PyTT pipelines work only slightly differently from spaCy's default pipelines.
Specifically, we introduce a new pipeline component at the start of the pipeline,
`PyTT_TokenVectorEncoder`. We then modify the [`nlp.update`](https://spacy.io/api/language#update) function to run
the `PyTT_TokenVectorEncoder` before the other pipeline components, and
backprop it after the other components are done.

#### <kbd>staticmethod</kbd> `PyTT_Language.install_extensions`

Register the [custom extension attributes](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Doc`, `Span` and `Token` objects. If the extensions have already been registered, spaCy will raise an error. The following extensions will be set:

#### <kbd>method</kbd> `PyTT_Language.make_doc`

Create a `Doc` object from text. Applies spaCy's tokenizer and the PyTorch-Transformers tokenizer and aligns the tokens.

| Name        | Type               | Description          |
| ----------- | ------------------ | -------------------- |
| `text`      | unicode            | The text to process. |
| **RETURNS** | `spacy.tokens.Doc` | The processed `Doc`. |

#### <kbd>method</kbd> `PyTT_Language.update`

Update the models in the pipeline.

| Name            | Type     | Description                                                                                                                                |
| --------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `docs`          | iterable | A batch of `Doc` objects or unicode. If unicode, a `Doc` object will be created from the text.                                             |
| `golds`         | iterable | A batch of `GoldParse` objects or dictionaries. Dictionaries will be used to create [`GoldParse`](https://spacy.io/api/goldparse) objects. |
| `drop`          | float    | The dropout rate.                                                                                                                          |
| `sgd`           | callable | An optimizer.                                                                                                                              |
| `losses`        | dict     | Dictionary to update with the loss, keyed by pipeline component.                                                                           |
| `component_cfg` | dict     | Config parameters for specific pipeline components, keyed by component name.                                                               |

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
