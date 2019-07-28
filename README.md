<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy wrapper for PyTorch Transformers

This package provides [spaCy](https://spacy.io) model pipelines that wrap
[HuggingFace's `pytorch-transformers`](https://github.com/huggingface/pytorch-transformers)
package, so you can use them in spaCy. The result is convenient access to
state-of-the-art transformer architectures, such as BERT, GPT2, XLNet, etc.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/11/master.svg?logo=azure-devops&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=11)
[![PyPi](https://img.shields.io/pypi/v/spacy-pytorch-transformers.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-pytorch-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-pytorch-transformers/all.svg?style=flat-square)](https://github.com/explosion/spacy-pytorch-transformers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## Features

-   Aligned tokenization.
-   Transfer learning, Text classification.
-   Fine-tuning.
-   Built-in hooks for context-sensitive vectors and similarity.
-   Out-of-the-box serialization.

## 🚀 Quickstart

Installing the package from pip will automatically install all dependencies,
including PyTorch and spaCy.

```bash
pip install spacy-pytorch-transformers
```

We've also pre-packaged the `bert-base-uncased` model as a spaCy model package
(~1gb). You can either use the `spacy download` command, or download the package
from the [model releases](#).

```bash
python -m spacy download en_bert_base_uncased_xl
```

Once the model is installed, you can load it in spaCy like any other model
package.

```python
import spacy

nlp = spacy.load("en_bert_base_uncased_xl")
doc = nlp("The dog barked. The puppy barked.")
print(doc[0:4].similarity(doc[4:8]))
```

## 📖 Usage

### Vectors and similarity

The `PyTT_TokenVectorEncoder` component of the model sets custom hooks that
override the default behaviour of the `.vector` attribute and `.similarity`
method of the `Token`, `Span` and `Doc` objects. By default, these usually refer
to the word vectors table at `nlp.vocab.vectors`. Naturally, in the transformer
models we'd rather use the `doc.tensor` attribute, since it holds a much more
informative context-sensitive representation.

```python
doc_company = nlp("Apple shares rose sharply on the news.")
doc_fruit = nlp("I was in Corsica when I learned this fantastic reciple for apple pie.")
apple_co = doc_company[0]
apple_fruit = doc_fruit[-3]
print(apple_co.similarity(nlp("fruit")))
print(apple_fruit.similarity(nlp("company")))
```

### Transfer learning

The main use case for pretrained transformer models is transfer learning. You
load in the pretrained weights, and start training on your data. This package
has custom pipeline components that make this especially easy.

The `pytt_textcat` component is based on spaCy's built-in
[`TextCategorizer`](https://spacy.io/api/textcategorizer) and supports using the
features assigned by the PyTorch-Transformers models via the token vector
encoder. You can use a pre-trained transformer model as the base model, add it
on top and train it like any other spaCy text classifier.

```python
TRAIN_DATA = [
    ("text1", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}})
]
DEV_DATA = [
    ("text2", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}})
]
```

```python
import spacy
from spacy.util import minibatch
import random

nlp = spacy.load("en_bert_base_uncased_xl")

textcat = nlp.create_pipe("pytt_textcat", config={"exclusive_classes": True})
for label in ("POSITIVE", "NEGATIVE"):
    textcat.add_label(label)
nlp.add_pipe(textcat)

optimizer = nlp.resume_training()
for i in range(10):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for batch in minibatch(TRAIN_DATA, size=128):
        texts, cats = zip(*batch)
        nlp.update(texts, cats, optimizer=optimizer, losses=losses)
    scores = nlp.evaluate(DEV_DATA)
    print(i, scores, losses)
nlp.to_disk("/bert-textcat")
```

### Serialization

Saving and loading pre-trained transformer models and packaging them as spaCy
models ✨just works ✨ (at least, it should). The wrapper and components follow
spaCy's API, so when you save and load the `nlp` object, it...

-   Writes the pre-trained weights to disk / bytes and loads them back in.
-   Adds `"lang_factory": "pytt"` in the `meta.json` so spaCy knows how to
    initialize the `Language` class when you load the model.
-   Adds this package and its version to the `"requirements"` in the
    `meta.json`, so when you run
    [`spacy package`](https://spacy.io/api/cli#package) to create an installable
    Python package it's automatically added to the setup's `install_requires`.

For example, if you've trained your own text classifier, you can package it like
this:

```bash
python -m spacy package /bert-textcat /output
cd /output/en_bert_base_uncased_xl-0.0.0
python setup.py sdist
pip install dist/en_bert_base_uncased_xl.tar.gz
```

### Extension attributes

This wrapper sets the following
[custom extension attributes](https://spacy.io/usage/processing-pipelines#custom-components-attributes)
on the `Doc`, `Span` and `Token` objects:

| Name                   | Type         | Description                                                                                                                    |
| ---------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `._.pytt_alignment`    | list         | Alignment between word-pieces and spaCy tokens. Contains lists of word-piece token indices (one per spaCy token).              |
| `._.pytt_word_pieces`  | list         | The word-piece IDs.                                                                                                            |
| `._.pytt_word_pieces_` | list         | The string forms of the word-piece IDs.                                                                                        |
| `._.pytt_outputs`      | `namedtuple` | All outputs produced by the PyTorch Transformer model.                                                                         |
| `._.pytt_gradients`    | `namedtuple` | Gradients of the pytt_outputs. These get incremented during `nlp.update`, and then cleared at the end once the update is made. |

The values can be accessed via the `._` attribute. For example:

```python
doc = nlp("This is a text.")
print(doc._.pytt_word_pieces)
```

### Setting up the pipeline

In order to run, the `nlp` object created using `PyTT_Language` requires two
components to run in order: the `PyTT_WordPiecer`, which assigns the word piece
tokens and the `PyTT_TokenVectorEncoder`, which assigns the token vectors. The
`pytt_name` argument defines the name of the pre-trained model to use.

```python
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder

name = "bert-base-uncased"
nlp = PyTT_Language(pytt_name=name, meta={"lang": "en"})
wordpiecer = PyTT_WordPiecer(nlp.vocab, pytt_name=name)
tok2vec = PyTT_TokenVectorEncoder(nlp.vocab, pytt_name=name).from_pretrained(nlp.vocab, name)
nlp.add_pipe(wordpiecer)
nlp.add_pipe(tok2vec)
print(nlp.pipe_names)  # ['pytt_wordpiecer', 'pytt_tok2vec']
```

### Tokenization alignment

Transformer models are usually trained on text preprocessed with the "word
piece" algorithm, which limits the number of distinct tokens the model needs to
consider. Word-piece is convenient for training neural networks, but it doesn't
produce segmentations that match up to any linguistic notion of a "word". Most
rare words will map to multiple word-piece tokens, and occassionally the
alignment will be many-to-many. `spacy-pytorch-transformers` calculates this
alignment, which you can access at `doc._.pytt_alignment`. It's a list of length
equal to the number of spaCy tokens. Each value in the list is a list of
consecutive integers, which are indexes into the word-pieces list.

If you can work on representations that aren't aligned to actual words, it's
best to use the raw outputs of the transformer, which can be accessed at
`doc._.pytt_outputs.last_hidden_state`. This variable gives you a tensor with
one row per word-piece token.

If you're working on token-level tasks such as part-of-speech tagging or
spelling correction, you'll want to work on the token-aligned features, which
are stored in the `doc.tensor` variable.

We've taken care to calculate the aligned `doc.tensor` representation as
faithfully as possible. When one spaCy token aligns against several word-piece
tokens, the token's vector will be a sum of the relevant slice of the tensor,
weighted by how many other spaCy tokens are aligned against that row. By using a
weighted sum, we ensure that the sum of the `doc.tensor` variable corresponds to
the sum of the raw `doc._.pytt_outputs.last_hidden_state[1:1]` values. The only
information missing from the `doc.tensor` are the vectors for the boundary
tokens. However, note that in many tasks, the vectors for the boundary tokens
are quite important (see e.g. the analysis of BERT's attention by Clark et al.
(2019), who found that these tokens are very often attended to).

### Batching, padding and per-sentence processing

Transformer models have cubic runtime and memory complexity with respect to
sequence length. This means that longer texts need to be divided into sentences
in order to achieve reasonable efficiency.

`spacy-pytorch-transformers` handles this internally, so long as some sort of
sentence-boundary detection component has been added to the pipeline. We
recommend:

```python
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, first=True)
```

The default rules for the sentencizer component are very simple, but you can
also create a custom sentence-boundary detection component that works well on
your data. See spaCy's documentation for details. If a sentencizer is available
and the `per_sentence=True` configuration option is set, the transformer model
will predict over sentences, and the resulting tensor features will be
reconstructed to produce document-level annotations.

In order to further improve efficiency, especially for CPU processing,
`spacy-pytorch-transformers` also performs length-based subbatching internally.
The subbatching regroups batches by sequence length, to minimise the amount of
padding required. The configuration option `batch_by_length` controls this
behaviour. You can set it to 0 to disable the subbatching, or set it to an
integer to require that the subbatches must be at least N sequences long.

The subbatching and per-sentence processing are used instead of input
truncation, which many transformer implementations otherwise resort to.
Truncating inputs is usually bad, as it results in the loss of arbitrary
information.

## 🎛 API

### <kbd>class</kbd> `PyTT_Language`

A subclass of [`Language`](https://spacy.io/api/language) that holds a
PyTorch-Transformer (PyTT) pipeline. PyTT pipelines work only slightly
differently from spaCy's default pipelines. Specifically, we introduce a new
pipeline component at the start of the pipeline, `PyTT_TokenVectorEncoder`. We
then modify the [`nlp.update`](https://spacy.io/api/language#update) function to
run the `PyTT_TokenVectorEncoder` before the other pipeline components, and
backprop it after the other components are done.

#### <kbd>staticmethod</kbd> `PyTT_Language.install_extensions`

Register the
[custom extension attributes](https://spacy.io/usage/processing-pipelines#custom-components-attributes)
on the `Doc`, `Span` and `Token` objects. If the extensions have already been
registered, spaCy will raise an error. [See here](#extension-attributes) for the
extension attributes that will be set. You shouldn't have to call this method
yourself – it already runs when you import the package.

#### <kbd>method</kbd> `PyTT_Language.__init__`

See [`Language.__init__`](https://spacy.io/api/language#init). Expects either a
`pytt_name` setting in the `meta` or as a keyword argument, specifying the
pre-trained model name. This is used to set up the model-specific tokenizer.

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

### <kbd>class</kbd> `PyTT_WordPiecer`

spaCy pipeline component to assign PyTorch-Transformers word-piece tokenization
to the Doc, which can then be used by the token vector encoder. Note that this
component doesn't modify spaCy's tokenization. It only sets extension attributes
`pytt_word_pieces_`, `pytt_word_pieces` and `pytt_alignment` (alignment between
word-piece tokens and spaCy tokens).

The component is available as `pytt_wordpiecer` and registered via an entry
point, so it can also be created using
[`nlp.create_pipe`](https://spacy.io/api/language#create_pipe):

```python
wordpiecer = nlp.create_pipe("wordpiecer")
```

#### Config

The component can be configured with the following settings, usually passed in
as the `**cfg`.

| Name        | Type    | Description                                            |
| ----------- | ------- | ------------------------------------------------------ |
| `pytt_name` | unicode | Name of pre-trained model, e.g. `"bert-base-uncased"`. |

#### <kbd>method</kbd> `PyTT_WordPiecer.__init__`

Initialize the component.

| Name        | Type                | Description                                            |
| ----------- | ------------------- | ------------------------------------------------------ |
| `vocab`     | `spacy.vocab.Vocab` | The spaCy vocab to use.                                |
| `name`      | unicode             | Name of pre-trained model, e.g. `"bert-base-uncased"`. |
| `**cfg`     | -                   | Optional config parameters.                            |
| **RETURNS** | `PyTT_WordPiecer`   | The word piecer.                                       |

#### <kbd>method</kbd> `PyTT_WordPiecer.predict`

Run the word-piece tokenizer on a batch of docs and return the extracted
strings.

| Name        | Type     | Description                                                                      |
| ----------- | -------- | -------------------------------------------------------------------------------- |
| `docs`      | iterable | A batch of `Doc`s to process.                                                    |
| **RETURNS** | tuple    | A `(strings, None)` tuple. The strings are lists of strings, one list per `Doc`. |

#### <kbd>method</kbd> `PyTT_WordPiecer.set_annotations`

Assign the extracted tokens and IDs to the `Doc` objects.

| Name      | Type     | Description               |
| --------- | -------- | ------------------------- |
| `docs`    | iterable | A batch of `Doc` objects. |
| `outputs` | iterable | A batch of outputs.       |

### <kbd>class</kbd> `PyTT_TokenVectorEncoder`

spaCy pipeline component to use PyTorch-Transformers models. The component
assigns the output of the transformer to the `doc._.pytt_outputs` extension
attribute. We also calculate an alignment between the word-piece tokens and the
spaCy tokenization, so that we can use the last hidden states to set the
`doc.tensor` attribute. When multiple word-piece tokens align to the same spaCy
token, the spaCy token receives the sum of their values.

The component is available as `pytt_tok2vec` and registered via an entry point,
so it can also be created using
[`nlp.create_pipe`](https://spacy.io/api/language#create_pipe):

```python
tok2vec = nlp.create_pipe("pytt_tok2vec")
```

#### Config

The component can be configured with the following settings, usually passed in
as the `**cfg`.

| Name              | Type    | Description                                                                                                                            |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `pytt_name`       | unicode | Name of pre-trained model, e.g. `"bert-base-uncased"`.                                                                                 |
| `batch_by_length` | int     | Minimum batch size for grouping texts into subbatches based on their length to reduce padding. Set to `0` to disable. Defaults to `1`. |
| `per_sentence`    | bool    | Apply the model over sentences using the `doc.sents` attribute.                                                                        |

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.from_pretrained`

Create a `PyTT_TokenVectorEncoder` instance using pre-trained weights from a
PyTorch-Transformers model, even if it's not installed as a spaCy package.

```python
from spacy_pytorch_transformers import PyTT_TokenVectorEncoder
from spacy.tokens import Vocab
tok2vec = PyTT_TokenVectorEncoder.from_pretrained(Vocab(), "bert-base-uncased")
```

| Name        | Type                      | Description                                            |
| ----------- | ------------------------- | ------------------------------------------------------ |
| `vocab`     | `spacy.vocab.Vocab`       | The spaCy vocab to use.                                |
| `name`      | unicode                   | Name of pre-trained model, e.g. `"bert-base-uncased"`. |
| `**cfg`     | -                         | Optional config parameters.                            |
| **RETURNS** | `PyTT_TokenVectorEncoder` | The token vector encoder.                              |

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.Model`

Create an instance of `PyTT_Wrapper`, which holds the PyTorch-Transformers
model.

| Name        | Type                 | Description                                            |
| ----------- | -------------------- | ------------------------------------------------------ |
| `name`      | unicode              | Name of pre-trained model, e.g. `"bert-base-uncased"`. |
| `**cfg`     | -                    | Optional config parameters.                            |
| **RETURNS** | `thinc.neural.Model` | The wrapped model.                                     |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.__init__`

Initialize the component.

| Name        | Type                          | Description                                             |
| ----------- | ----------------------------- | ------------------------------------------------------- |
| `vocab`     | `spacy.vocab.Vocab`           | The spaCy vocab to use.                                 |
| `model`     | `thinc.neural.Model` / `True` | The component's model or `True` if not initialized yet. |
| `**cfg`     | -                             | Optional config parameters.                             |
| **RETURNS** | `PyTT_TokenVectorEncoder`     | The token vector encoder.                               |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.__call__`

Process a `Doc` and assign the extracted features.

| Name        | Type               | Description           |
| ----------- | ------------------ | --------------------- |
| `doc`       | `spacy.tokens.Doc` | The `Doc` to process. |
| **RETURNS** | `spacy.tokens.Doc` | The processed `Doc`.  |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.pipe`

Process `Doc` objects as a stream and assign the extracted features.

| Name         | Type               | Description                                       |
| ------------ | ------------------ | ------------------------------------------------- |
| `stream`     | iterable           | A stream of `Doc` objects.                        |
| `batch_size` | int                | The number of texts to buffer. Defaults to `128`. |
| **YIELDS**   | `spacy.tokens.Doc` | Processed `Doc`s in order.                        |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.predict`

Run the transformer model on a batch of docs and return the extracted features.

| Name        | Type         | Description                         |
| ----------- | ------------ | ----------------------------------- |
| `docs`      | iterable     | A batch of `Doc`s to process.       |
| **RETURNS** | `namedtuple` | Named tuple containing the outputs. |

#### <kbd>method</kbd> `PyTT_TokenVectorEncoder.set_annotations`

Assign the extracted features to the Doc objects and overwrite the vector and
similarity hooks.

| Name      | Type     | Description               |
| --------- | -------- | ------------------------- |
| `docs`    | iterable | A batch of `Doc` objects. |
| `outputs` | iterable | A batch of outputs.       |

### <kbd>class</kbd> `PyTT_TextCategorizer`

Subclass of spaCy's built-in
[`TextCategorizer`](https://spacy.io/api/textcategorizer) component that
supports using the features assigned by the PyTorch-Transformers models via the
token vector encoder. It requires the `PyTT_TokenVectorEncoder` to run before it
in the pipeline.

The component is available as `pytt_textcat` and registered via an entry point,
so it can also be created using
[`nlp.create_pipe`](https://spacy.io/api/language#create_pipe):

```python
textcat = nlp.create_pipe("pytt_textcat")
```

#### <kbd>classmethod</kbd> `PyTT_TextCategorizer.Model`

Create a text classification model using a PyTorch-Transformers model for token
vector encoding.

| Name                | Type                 | Description                                              |
| ------------------- | -------------------- | -------------------------------------------------------- |
| `nr_class`          | int                  | Number of classes.                                       |
| `width`             | int                  | The width of the tensors being assigned.                 |
| `exclusive_classes` | bool                 | Make categories mutually exclusive. Defaults to `False`. |
| `**cfg`             | -                    | Optional config parameters.                              |
| **RETURNS**         | `thinc.neural.Model` | The model.                                               |

### Entry points

This package exposes several
[entry points](https://spacy.io/usage/saving-loading#entry-points) that tell
spaCy how to initialize its components. If `spacy-pytorch-transformers` and
spaCy are installed in the same environment, you'll be able to run the following
and it'll work as expected:

```python
tok2vec = nlp.create_pipe("pytt_tok2vec")
```

This also means that your custom models can ship a `pytt_tok2vec` component and
define `"pytt_tok2vec"` in their pipelines, and spaCy will know how to create
those components when you deserialize the model. The following entry points are
set:

| Name              | Target                    | Type              | Description                      |
| ----------------- | ------------------------- | ----------------- | -------------------------------- |
| `pytt_wordpiecer` | `PyTT_WordPiecer`         | `spacy_factories` | Factory to create the component. |
| `pytt_tok2vec`    | `PyTT_TokenVectorEncoder` | `spacy_factories` | Factory to create the component. |
| `pytt_textcat`    | `PyTT_TextCategorizer`    | `spacy_factories` | Factory to create the component. |
| `pytt`            | `PyTT_Language`           | `spacy_languages` | Custom `Language` subclass.      |
