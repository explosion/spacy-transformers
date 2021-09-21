<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-transformers: Use pretrained transformers like BERT, XLNet and GPT-2 in spaCy

This package provides [spaCy](https://github.com/explosion/spaCy) components and
architectures to use transformer models via
[Hugging Face's `transformers`](https://github.com/huggingface/transformers) in
spaCy. The result is convenient access to state-of-the-art transformer
architectures, such as BERT, GPT-2, XLNet, etc.

> **This release requires [spaCy v3](https://spacy.io/usage/v3).** For
> the previous version of this library, see the
> [`v0.6.x` branch](https://github.com/explosion/spacy-transformers/tree/v0.6.x).

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/18/master.svg?logo=azure-pipelines&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=18)
[![PyPi](https://img.shields.io/pypi/v/spacy-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-transformers)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-transformers/all.svg?style=flat-square&logo=github)](https://github.com/explosion/spacy-transformers/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## Features

- Use pretrained transformer models like **BERT**, **RoBERTa** and **XLNet** to
  power your spaCy pipeline.
- Easy **multi-task learning**: backprop to one transformer model from several
  pipeline components.
- Train using spaCy v3's powerful and extensible config system.
- Automatic alignment of transformer output to spaCy's tokenization.
- Easily customize what transformer data is saved in the `Doc` object.
- Easily customize how long documents are processed.
- Out-of-the-box serialization and model packaging.

## 🚀 Installation

Installing the package from pip will automatically install all dependencies,
including PyTorch and spaCy. Make sure you install this package **before** you
install the models. Also note that this package requires **Python 3.6+**,
**PyTorch v1.5+** and **spaCy v3.0+**.

```bash
pip install spacy[transformers]
```

For GPU installation, find your CUDA version using `nvcc --version` and add the
[version in brackets](https://spacy.io/usage/#gpu), e.g.
`spacy[transformers,cuda92]` for CUDA9.2 or `spacy[transformers,cuda100]` for
CUDA10.0.

If you are having trouble installing PyTorch, follow the
[instructions](https://pytorch.org/get-started/locally/) on the official website
for your specific operating system and requirements, or try the following:

```bash
pip install spacy-transformers -f https://download.pytorch.org/whl/torch_stable.html
```

## 📖 Documentation

> ⚠️ **Important note:** This package has been extensively refactored to take
> advantage of [spaCy v3.0](https://spacy.io). Previous versions that
> were built for [spaCy v2.x](https://v2.spacy.io) worked considerably
> differently. Please see previous tagged versions of this README for
> documentation on prior versions.

- 📘
  [Embeddings, Transformers and Transfer Learning](https://spacy.io/usage/embeddings-transformers):
  How to use transformers in spaCy
- 📘 [Training Pipelines and Models](https://spacy.io/usage/training):
  Train and update components on your own data and integrate custom models
- 📘
  [Layers and Model Architectures](https://spacy.io/usage/layers-architectures):
  Power spaCy components with custom neural networks
- 📗 [`Transformer`](https://spacy.io/api/transformer): Pipeline
  component API reference
- 📗
  [Transformer architectures](https://spacy.io/api/architectures#transformers):
  Architectures and registered functions
