<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-transformers

This package (previously `spacy-pytorch-transformers`) provides
[spaCy](https://github.com/explosion/spaCy) model pipelines that wrap
[Hugging Face's `transformers`](https://github.com/huggingface/transformers)
package, so you can use them in spaCy. The result is convenient access to
state-of-the-art transformer architectures, such as BERT, GPT-2, XLNet, etc.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/18/master.svg?logo=azure-pipelines&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=18)
[![PyPi](https://img.shields.io/pypi/v/spacy-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-transformers/all.svg?style=flat-square&logo=github)](https://github.com/explosion/spacy-transformers/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![Open demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/explosion/spacy-transformers/blob/master/examples/Spacy_Transformers_Demo.ipynb)

## Features

-   Use pretrained transformer models like **BERT**, **RoBERTa** and **XLNet** to
    power your spaCy pipeline.
-   Easy **multi-task learning**: backprop to one transformer model from
    several pipeline components.
-   Train using spaCy v3's powerful and extensible config system.
-   Automatic alignment of transformer output to spaCy's tokenization.
-   Easily customize what transformer data is saved in the `Doc` object.
-   Easily customize how long documents are processed.
-   Out-of-the-box serialization and model packaging.

## 🚀 Installation

Installing the package from pip will automatically install all dependencies,
including PyTorch and spaCy. Make sure you install this package **before** you
install the models. Also note that this package requires **Python 3.6+** and
spaCy v3.

```bash
pip install spacy-transformers
```

For GPU installation, find your CUDA version using `nvcc --version` and add the
[version in brackets](https://spacy.io/usage/#gpu), e.g.
`spacy-transformers[cuda92]` for CUDA9.2 or `spacy-transformers[cuda100]` for
CUDA10.0.

If you are having trouble installing PyTorch, follow the 
[instructions](https://pytorch.org/get-started/locally/) on the official website 
for your specific operation system and requirements.

## 📖 Usage

> ⚠️ **Important note:** This package has been extensively refactored to take
> advantage of spaCy v3. Previous versions that were built for spaCy v2 worked
> considerably differently. Please see previous tagged versions of this readme
> for documentation on prior versions.

spaCy v3 lets you use almost any statistical model to power your pipeline. You
can use models implemented in a variety of frameworks, including Tensorflow,
PyTorch and MXNet. To keep things sane, spaCy expects models from these
frameworks to be wrapped with a common interface, using our machine learning
library [Thinc](https://thinc.ai). A transformer model is just a statistical
model, so the `spacy-transformers` package actually has very little work to do:
we just have to provide a few functions that do the required plumbing. We also
provide a pipeline component, `Transformer`, that lets you do multi-task
learning and lets you save the transformer outputs for later use.

### Training usage

The recommended workflow for training is to use spaCy v3's new config system,
usually via the `spacy train-from-config` command-line command. See here for an
end-to-end example. The config system lets you describe a tree of objects by
referring to creation functions, including functions you register yourself.
Here's a config snippet for the `Transformer` component, along with matching
Python code.

```
[nlp.pipeline.transformer]
factory = "transformer"
set_extra_annotations = null
max_batch_size = 32

[nlp.pipeline.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v1"
name = "bert-base-cased"
tokenizer_config = {"use_fast": true}

[nlp.pipeline.transformer.model.get_spans]
@span_getters = "spacy-transformers.get_doc_spans.v1"
```

```python

trf = Transformer(
    nlp.vocab,
    TransformerModel(
        "bert-base-cased",
        get_spans=get_doc_spans,
        tokenizer_config={"use_fast": True},
    ),
    set_extra_annotations=null_annotation_setter,
    max_batch_size=32,
)
nlp.add_pipe("transformer", trf, first=True)
```

The `nlp.pipeline.transformer` block adds the `transformer` component to the
pipeline, and the `nlp.pipeline.transformer.model` block describes the creation
of a Thinc `Model` object that will be passed into the component. The block
names a function registered in the `@architectures` registry. This function
will be looked up and called using the provided arguments. You're not limited
to just that function --- you can write your own or use someone else's. The
only limitation is that it must return an object of type `Model[List[Doc],
FullTransformerBatch]`: that is, a Thinc model that takes a list of `Doc`
objects, and returns a `FullTransformerBatch` object with the transformer data.

The same idea applies to task models that power the downstream components.
Most of spaCy's built-in model creation functions support a `tok2vec` argument,
which should be a Thinc layer of type `Model[List[Doc], List[Floats2d]]`. This
is where we'll plug in our transformer model, using the `Tok2VecTransformer`
layer, which sneakily delegates to the `Transformer` pipeline component.

```
[nlp.pipeline.ner]
factory = "ner"

[nlp.pipeline.ner.model]
@architectures = "spacy.TransitionBasedParser.v1"
nr_feature_tokens = 3
hidden_width = 128
maxout_pieces = 3
use_upper = false

[nlp.pipeline.ner.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[nlp.pipeline.ner.model.tok2vec.pooling]
@layers = "reduce_mean.v1"
```

The `TransformerListener` layer expects a `pooling` layer, which needs
to be of type `Model[Ragged, Floats2d]`. This layer determines how the vector
for each spaCy token will be computed from the zero or more source rows the
token is aligned against. Here we use the `reduce_mean` layer, which averages
the wordpiece rows. We could instead use `reduce_last`, `reduce_max`, or
a custom function you write yourself.

You can have multiple components all listening to the same transformer model,
and all passing gradients back to it. By default, all of the gradients will
be equally weighted. You can control this with the `grad_factor` setting,
which lets you reweight the gradients from the different listeners. For
instance, setting `grad_factor = 0` would disable gradients from one of the
listeners, while `grad_factor = 2.0` would multiply them by 2. This is similar
to having a custom learning rate for each component. Instead of a constant, you
can also provide a schedule, allowing you to freeze the shared parameters at
the start of training.

### Runtime usage

Transformer models can be used as drop-in replacements for other types of
neural networks, so your spaCy pipeline can include them in a way that's
completely invisible to the user. Users will download, load and use the model
in the standard way, like any other spaCy pipeline.

Instead of using the transformers as subnetworks directly, you can also use them
via the `Transformer` pipeline component. This sets the `doc._.trf_data` extension
attribute, which lets you access the transformers outputs at runtime via the
`doc._.trf_data` extension attribute. You can also customize how the
`Transformer` object sets annotations onto the `Doc`, by customizing the 
`Transformer.set_extra_annotations` function. This callback will be called with the
raw input and output data for the whole batch, along with the batch of `Doc`
objects, allowing you to implement whatever you need.

```python

import spacy

nlp = spacy.load("en_core_trf_lg")
for doc in nlp.pipe(["some text", "some other text"]):
    doc._.trf_data.tensors
    tokvecs = doc._.trf_data.tensors[-1]
```

The `nlp` object in this example is just like any other spaCy pipeline, so
see the [spaCy docs](https://spacy.io/usage/processing-pipelines) for more details 
about what you can do.
