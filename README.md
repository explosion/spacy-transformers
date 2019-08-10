<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy wrapper for PyTorch Transformers

This package provides [spaCy](https://github.com/explosion/spaCy) model
pipelines that wrap
[Hugging Face's `pytorch-transformers`](https://github.com/huggingface/pytorch-transformers)
package, so you can use them in spaCy. The result is convenient access to
state-of-the-art transformer architectures, such as BERT, GPT-2, XLNet, etc. For
more details and background, check out
[our blog post](https://explosion.ai/blog/spacy-pytorch-transformers).

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/11/master.svg?logo=azure-devops&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=11)
[![PyPi](https://img.shields.io/pypi/v/spacy-pytorch-transformers.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-pytorch-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-pytorch-transformers/all.svg?style=flat-square)](https://github.com/explosion/spacy-pytorch-transformers/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## Features

-   Use **BERT**, **XLNet** and **GPT-2** directly in your spaCy pipeline.
-   **Fine-tune** pretrained transformer models on your task using spaCy's API.
-   Custom component for **text classification** using transformer features.
-   Automatic **alignment** of wordpieces and outputs to linguistic tokens.
-   Process multi-sentence documents with intelligent **per-sentence
    prediction**.
-   Built-in hooks for **context-sensitive vectors** and similarity.
-   Out-of-the-box serialization and model packaging.

## 🚀 Quickstart

Installing the package from pip will automatically install all dependencies,
including PyTorch and spaCy. Make sure you install this package **before** you
install the models. Also note that this package requires **Python 3.6+** and the
latest version of spaCy,
[v2.1.7](https://github.com/explosion/spaCy/releases/tag/v2.1.7) or above.

```bash
pip install spacy-pytorch-transformers
```

For GPU installation, find your CUDA version using `nvcc --version` and add the
[version in brackets](https://spacy.io/usage/#gpu), e.g.
`spacy-pytorch-transformers[cuda92]` for CUDA9.2 or
`spacy-pytorch-transformers[cuda100]` for CUDA10.0.

We've also pre-packaged some of the pretrained models as spaCy model packages.
You can either use the `spacy download` command or download the packages from
the [model releases](https://github.com/explosion/spacy-models/releases).

| Package name                 | Pretrained model         | Language | Author                                                     |  Size |                                            Release                                             |
| ---------------------------- | ------------------------ | -------- | ---------------------------------------------------------- | ----: | :--------------------------------------------------------------------------------------------: |
| `en_pytt_bertbaseuncased_lg` | `bert-base-uncased`      | English  | [Google Research](https://github.com/google-research/bert) | 406MB | [📦️](https://github.com/explosion/spacy-models/releases/tag/en_pytt_bertbaseuncased_lg-2.1.1) |
| `de_pytt_bertbasecased_lg`   | `bert-base-german-cased` | German   | [deepset](https://deepset.ai/german-bert)                  | 406MB |  [📦️](https://github.com/explosion/spacy-models/releases/tag/de_pytt_bertbasecased_lg-2.1.1)  |
| `en_pytt_xlnetbasecased_lg`  | `xlnet-base-cased`       | English  | [CMU/Google Brain](https://github.com/zihangdai/xlnet/)    | 434MB | [📦️](https://github.com/explosion/spacy-models/releases/tag/en_pytt_xlnetbasecased_lg-2.1.1)  |

```bash
python -m spacy download en_pytt_bertbaseuncased_lg
python -m spacy download de_pytt_bertbasecased_lg
python -m spacy download en_pytt_xlnetbasecased_lg
```

Once the model is installed, you can load it in spaCy like any other model
package.

```python
import spacy

nlp = spacy.load("en_pytt_bertbaseuncased_lg")
doc = nlp("Apple shares rose on the news. Apple pie is delicious.")
print(doc[0].similarity(doc[7]))
print(doc._.pytt_last_hidden_state.shape)
```

> 💡 If you're seeing an error like `No module named 'spacy.lang.pytt'`,
> double-check that `spacy-pytorch-transformers` is installed. It needs to be
> available so it can register its language entry points. Also make sure that
> you're running spaCy v2.1.7 or higher.

## 📖 Usage

### Transfer learning

The main use case for pretrained transformer models is transfer learning. You
load in a large generic model pretrained on lots of text, and start training on
your smaller dataset with labels specific to your problem. This package has
custom pipeline components that make this especially easy. We provide an example
component for text categorization. Development of analogous components for other
tasks should be quite straight-forward.

The `pytt_textcat` component is based on spaCy's built-in
[`TextCategorizer`](https://spacy.io/api/textcategorizer) and supports using the
features assigned by the PyTorch-Transformers models, via the `pytt_tok2vec`
component. This lets you use a model like BERT to predict contextual token
representations, and then learn a text categorizer on top as a task-specific
"head". The API is the same as any other spaCy pipeline:

```python
TRAIN_DATA = [
    ("text1", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}})
]
```

```python
import spacy
from spacy.util import minibatch
import random
import torch

is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

nlp = spacy.load("en_pytt_bertbaseuncased_lg")
print(nlp.pipe_names) # ["sentencizer", "pytt_wordpiecer", "pytt_tok2vec"]
textcat = nlp.create_pipe("pytt_textcat", config={"exclusive_classes": True})
for label in ("POSITIVE", "NEGATIVE"):
    textcat.add_label(label)
nlp.add_pipe(textcat)

optimizer = nlp.resume_training()
for i in range(10):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for batch in minibatch(TRAIN_DATA, size=8):
        texts, cats = zip(*batch)
        nlp.update(texts, cats, sgd=optimizer, losses=losses)
    print(i, losses)
nlp.to_disk("/bert-textcat")
```

For a full example, see the
[`examples/train_textcat.py` script](examples/train_textcat.py).

### Vectors and similarity

The `PyTT_TokenVectorEncoder` component of the model sets custom hooks that
override the default behaviour of the `.vector` attribute and `.similarity`
method of the `Token`, `Span` and `Doc` objects. By default, these usually refer
to the word vectors table at `nlp.vocab.vectors`. Naturally, in the transformer
models we'd rather use the `doc.tensor` attribute, since it holds a much more
informative context-sensitive representation.

```python
apple1 = nlp("Apple shares rose on the news.")
apple2 = nlp("Apple sold fewer iPhones this quarter.")
apple3 = nlp("Apple pie is delicious.")
print(apple1[0].similarity(apple2[0]))
print(apple1[0].similarity(apple3[0]))
```

### Serialization

Saving and loading pretrained transformer models and packaging them as spaCy
models ✨just works ✨ (at least, it should). The wrapper and components follow
spaCy's API, so when you save and load the `nlp` object, it...

-   Writes the pretrained weights to disk / bytes and loads them back in.
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
cd /output/en_pytt_bertbaseuncased_lg-1.0.0
python setup.py sdist
pip install dist/en_pytt_bertbaseuncased_lg-1.0.0.tar.gz
```

### Extension attributes

This wrapper sets the following
[custom extension attributes](https://spacy.io/usage/processing-pipelines#custom-components-attributes)
on the `Doc`, `Span` and `Token` objects:

| Name                          | Type              | Description                                                                                                                                                   |
| ----------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `._.pytt_alignment`           | `List[List[int]]` | Alignment between wordpieces and spaCy tokens. Contains lists of wordpiece token indices (one per spaCy token) or a list of indices (if called on a `Token`). |
| `._.pytt_word_pieces`         | `List[int]`       | The wordpiece IDs.                                                                                                                                            |
| `._.pytt_word_pieces_`        | `List[str]`       | The string forms of the wordpiece IDs.                                                                                                                        |
| `._.pytt_last_hidden_state`   | `ndarray`         | The `last_hidden_state` output from the PyTorch-Transformers model.                                                                                           |
| `._.pytt_pooler_output`       | `List[ndarray]`   | The `pooler_output` output from the PyTorch-Transformers model.                                                                                               |
| `._.pytt_all_hidden_states`   | `List[ndarray]`   | The `all_hidden_states` output from the PyTorch-Transformers model.                                                                                           |
| `._.all_attentions`           | `List[ndarray]`   | The `all_attentions` output from the PyTorch-Transformers model.                                                                                              |
| `._.pytt_d_last_hidden_state` | `ndarray`         | The gradient of the `last_hidden_state` output from the PyTorch-Transformers model.                                                                           |
| `._.pytt_d_pooler_output`     | `List[ndarray]`   | The gradient of the `pooler_output` output from the PyTorch-Transformers model.                                                                               |
| `._.pytt_d_all_hidden_states` | `List[ndarray]`   | The gradient of the `all_hidden_states` output from the PyTorch-Transformers model.                                                                           |
| `._.pytt_d_all_attentions`    | `List[ndarray]`   | The gradient of the `all_attentions` output from the PyTorch-Transformers model.                                                                              |

The values can be accessed via the `._` attribute. For example:

```python
doc = nlp("This is a text.")
print(doc._.pytt_word_pieces_)
```

### Setting up the pipeline

In order to run, the `nlp` object created using `PyTT_Language` requires a few
components to run in order: a component that assigns sentence boundaries (e.g.
spaCy's built-in
[`Sentencizer`](https://spacy.io/usage/linguistic-features#sbd-component)), the
`PyTT_WordPiecer`, which assigns the wordpiece tokens and the
`PyTT_TokenVectorEncoder`, which assigns the token vectors. The `pytt_name`
argument defines the name of the pretrained model to use. The `from_pretrained`
methods load the pretrained model via `pytorch-transformers`.

```python
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder

name = "bert-base-uncased"
nlp = PyTT_Language(pytt_name=name, meta={"lang": "en"})
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, name))
nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, name))
print(nlp.pipe_names)  # ['sentencizer', 'pytt_wordpiecer', 'pytt_tok2vec']
```

You can also use the [`init_model.py`](examples/init_model.py) script in the
examples.

#### Loading models from a path

Pytorch-Transformers models can also be loaded from a file path instead of just
a name. For instance, let's say you want to use Allen AI's
[`scibert`](https://github.com/allenai/scibert). First, download the PyTorch
model files, unpack them them, unpack the `weights.tar`, rename the
`bert_config.json` to `config.json` and put everything into one directory. Your
directory should now have a `pytorch_model.bin`, `vocab.txt` and `config.json`.
Also make sure that your path **includes the name of the model**. You can then
initialize the `nlp` object like this:

```python
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder

name = "scibert-scivocab-uncased"
path = "/path/to/scibert-scivocab-uncased"

nlp = PyTT_Language(pytt_name=name, meta={"lang": "en"})
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, path))
nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, path))
```

### Tokenization alignment

Transformer models are usually trained on text preprocessed with the "wordpiece"
algorithm, which limits the number of distinct token-types the model needs to
consider. Wordpiece is convenient for training neural networks, but it doesn't
produce segmentations that match up to any linguistic notion of a "word". Most
rare words will map to multiple wordpiece tokens, and occasionally the alignment
will be many-to-many. `spacy-pytorch-transformers` calculates this alignment,
which you can access at `doc._.pytt_alignment`. It's a list of length equal to
the number of spaCy tokens. Each value in the list is a list of consecutive
integers, which are indexes into the wordpieces list.

If you can work on representations that aren't aligned to actual words, it's
best to use the raw outputs of the transformer, which can be accessed at
`doc._.pytt_last_hidden_state`. This variable gives you a tensor with one row
per wordpiece token.

If you're working on token-level tasks such as part-of-speech tagging or
spelling correction, you'll want to work on the token-aligned features, which
are stored in the `doc.tensor` variable.

We've taken care to calculate the aligned `doc.tensor` representation as
faithfully as possible, with priority given to avoid information loss. The
alignment has been calculated such that
`doc.tensor.sum(axis=1) == doc._.pytt_last_hidden_state.sum(axis=1)`. To make
this work, each row of the `doc.tensor` (which corresponds to a spaCy token) is
set to a weighted sum of the rows of the `last_hidden_state` tensor that the
token is aligned to, where the weighting is proportional to the number of other
spaCy tokens aligned to that row. To include the information from the (often
important --- see Clark et al., 2019) boundary tokens, we imagine that these are
also "aligned" to all of the tokens in the sentence.

### Batching, padding and per-sentence processing

Transformer models have cubic runtime and memory complexity with respect to
sequence length. This means that longer texts need to be divided into sentences
in order to achieve reasonable efficiency.

`spacy-pytorch-transformers` handles this internally, and requires that sort of
sentence-boundary detection component has been added to the pipeline. We
recommend:

```python
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, first=True)
```

Internally, the transformer model will predict over sentences, and the resulting
tensor features will be reconstructed to produce document-level annotations.

In order to further improve efficiency and reduce memory requirements,
`spacy-pytorch-transformers` also performs length-based subbatching internally.
The subbatching regroups the batched sentences by sequence length, to minimise
the amount of padding required. The configuration option `words_per_batch`
controls this behaviour. You can set it to 0 to disable the subbatching, or set
it to an integer to require a maximum limit on the number of words (including
padding) per subbatch. The default value of 3000 words works reasonably well on
a Tesla V100.

Many of the pretrained transformer models have a maximum sequence length. If a
sentence is longer than the maximum, it is truncated and the affected ending
tokens will receive zeroed vectors.

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
pretrained model name. This is used to set up the model-specific tokenizer.

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

spaCy pipeline component to assign PyTorch-Transformers wordpiece tokenization
to the Doc, which can then be used by the token vector encoder. Note that this
component doesn't modify spaCy's tokenization. It only sets extension attributes
`pytt_word_pieces_`, `pytt_word_pieces` and `pytt_alignment` (alignment between
wordpiece tokens and spaCy tokens).

The component is available as `pytt_wordpiecer` and registered via an entry
point, so it can also be created using
[`nlp.create_pipe`](https://spacy.io/api/language#create_pipe):

```python
wordpiecer = nlp.create_pipe("wordpiecer")
```

#### Config

The component can be configured with the following settings, usually passed in
as the `**cfg`.

| Name        | Type    | Description                                           |
| ----------- | ------- | ----------------------------------------------------- |
| `pytt_name` | unicode | Name of pretrained model, e.g. `"bert-base-uncased"`. |

#### <kbd>classmethod</kbd> `PyTT_WordPiecer.from_nlp`

Factory to add to `Language.factories` via entry point.

| Name        | Type                      | Description                                     |
| ----------- | ------------------------- | ----------------------------------------------- |
| `nlp`       | `spacy.language.Language` | The `nlp` object the component is created with. |
| `**cfg`     | -                         | Optional config parameters.                     |
| **RETURNS** | `PyTT_WordPiecer`         | The wordpiecer.                                 |

#### <kbd>method</kbd> `PyTT_WordPiecer.__init__`

Initialize the component.

| Name        | Type                | Description                                           |
| ----------- | ------------------- | ----------------------------------------------------- |
| `vocab`     | `spacy.vocab.Vocab` | The spaCy vocab to use.                               |
| `name`      | unicode             | Name of pretrained model, e.g. `"bert-base-uncased"`. |
| `**cfg`     | -                   | Optional config parameters.                           |
| **RETURNS** | `PyTT_WordPiecer`   | The wordpiecer.                                       |

#### <kbd>method</kbd> `PyTT_WordPiecer.predict`

Run the wordpiece tokenizer on a batch of docs and return the extracted strings.

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
assigns the output of the transformer to extension attributes. We also calculate
an alignment between the wordpiece tokens and the spaCy tokenization, so that we
can use the last hidden states to set the `doc.tensor` attribute. When multiple
wordpiece tokens align to the same spaCy token, the spaCy token receives the sum
of their values.

The component is available as `pytt_tok2vec` and registered via an entry point,
so it can also be created using
[`nlp.create_pipe`](https://spacy.io/api/language#create_pipe):

```python
tok2vec = nlp.create_pipe("pytt_tok2vec")
```

#### Config

The component can be configured with the following settings, usually passed in
as the `**cfg`.

| Name              | Type    | Description                                                                                                                                                                                                                 |
| ----------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pytt_name`       | unicode | Name of pretrained model, e.g. `"bert-base-uncased"`.                                                                                                                                                                       |
| `words_per_batch` | int     | Group sentences into subbatches of max `words_per_batch` in size. For instance, a batch with one 100 word sentence and one 10 word sentence will have size 200 (due to padding). Set to `0` to disable. Defaults to `2000`. |

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.from_nlp`

Factory to add to `Language.factories` via entry point.

| Name        | Type                      | Description                                     |
| ----------- | ------------------------- | ----------------------------------------------- |
| `nlp`       | `spacy.language.Language` | The `nlp` object the component is created with. |
| `**cfg`     | -                         | Optional config parameters.                     |
| **RETURNS** | `PyTT_TokenVectorEncoder` | The token vector encoder.                       |

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.from_pretrained`

Create a `PyTT_TokenVectorEncoder` instance using pretrained weights from a
PyTorch-Transformers model, even if it's not installed as a spaCy package.

```python
from spacy_pytorch_transformers import PyTT_TokenVectorEncoder
from spacy.tokens import Vocab
tok2vec = PyTT_TokenVectorEncoder.from_pretrained(Vocab(), "bert-base-uncased")
```

| Name        | Type                      | Description                                           |
| ----------- | ------------------------- | ----------------------------------------------------- |
| `vocab`     | `spacy.vocab.Vocab`       | The spaCy vocab to use.                               |
| `name`      | unicode                   | Name of pretrained model, e.g. `"bert-base-uncased"`. |
| `**cfg`     | -                         | Optional config parameters.                           |
| **RETURNS** | `PyTT_TokenVectorEncoder` | The token vector encoder.                             |

#### <kbd>classmethod</kbd> `PyTT_TokenVectorEncoder.Model`

Create an instance of `PyTT_Wrapper`, which holds the PyTorch-Transformers
model.

| Name        | Type                 | Description                                           |
| ----------- | -------------------- | ----------------------------------------------------- |
| `name`      | unicode              | Name of pretrained model, e.g. `"bert-base-uncased"`. |
| `**cfg`     | -                    | Optional config parameters.                           |
| **RETURNS** | `thinc.neural.Model` | The wrapped model.                                    |

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

#### <kbd>classmethod</kbd> `PyTT_TextCategorizer.from_nlp`

Factory to add to `Language.factories` via entry point.

| Name        | Type                      | Description                                     |
| ----------- | ------------------------- | ----------------------------------------------- |
| `nlp`       | `spacy.language.Language` | The `nlp` object the component is created with. |
| `**cfg`     | -                         | Optional config parameters.                     |
| **RETURNS** | `PyTT_TextCategorizer`    | The text categorizer.                           |

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

### <kbd>dataclass</kbd> `Activations`

Dataclass to hold the features produced by PyTorch-Transformers.

| Attribute           | Type   | Description |
| ------------------- | ------ | ----------- |
| `last_hidden_state` | object |             |
| `pooler_output`     | object |             |
| `all_hidden_states` | object |             |
| `all_attentions`    | object |             |
| `is_grad`           | bool   |             |

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
