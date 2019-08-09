import os
import io
from setuptools import setup, find_packages


def setup_package():
    package_name = "spacy_pytorch_transformers"
    root = os.path.abspath(os.path.dirname(__file__))

    # Read in package meta from about.py
    about_path = os.path.join(root, package_name, "about.py")
    with io.open(about_path, encoding="utf8") as f:
        about = {}
        exec(f.read(), about)

    # Get readme
    readme_path = os.path.join(root, "README.md")
    with io.open(readme_path, encoding="utf8") as f:
        readme = f.read()

    setup(
        name="spacy-pytorch-transformers",
        description=about["__summary__"],
        long_description=readme,
        long_description_content_type="text/markdown",
        author=about["__author__"],
        author_email=about["__email__"],
        url=about["__uri__"],
        version=about["__version__"],
        license=about["__license__"],
        packages=find_packages(),
        install_requires=[
            "spacy>=2.1.7,<2.2.0",
            "pytorch_transformers>=1.0.0,<1.1.0",
            "torch>=1.0.0",
            "torchcontrib>=0.0.2,<0.1.0",
            "srsly>=0.0.7,<1.1.0",
            "ftfy>=5.0.0,<6.0.0",
            "dataclasses>=0.6,<0.7; python_version < '3.7'",
        ],
        python_requires=">=3.6",
        extras_require={
            "cuda": ["thinc_gpu_ops>=0.0.1,<0.1.0", "cupy>=5.0.0b4"],
            "cuda80": ["thinc_gpu_ops>=0.0.1,<0.1.0", "cupy-cuda80>=5.0.0b4"],
            "cuda90": ["thinc_gpu_ops>=0.0.1,<0.1.0", "cupy-cuda90>=5.0.0b4"],
            "cuda91": ["thinc_gpu_ops>=0.0.1,<0.1.0", "cupy-cuda91>=5.0.0b4"],
            "cuda92": ["thinc_gpu_ops>=0.0.1,<0.1.0", "cupy-cuda92>=5.0.0b4"],
            "cuda100": ["thinc_gpu_ops>=0.0.1,<0.1.0", "cupy-cuda100>=5.0.0b4"],
        },
        entry_points={
            "spacy_factories": [
                "pytt_wordpiecer = spacy_pytorch_transformers:PyTT_WordPiecer.from_nlp",
                "pytt_tok2vec = spacy_pytorch_transformers:PyTT_TokenVectorEncoder.from_nlp",
                "pytt_textcat = spacy_pytorch_transformers:PyTT_TextCategorizer.from_nlp",
            ],
            "spacy_languages": ["pytt = spacy_pytorch_transformers:PyTT_Language"],
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
