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
            "spacy>=2.1.6,<2.2.0",
            "pytorch_transformers>=1.0.0,<1.1.0",
            "torch>=0.4.1",
            "srsly>=0.0.7,<1.1.0",
        ],
        python_requires=">=3.6",
        entry_points={
            "spacy_factories": [
                "pytt_wordpiecer = spacy_pytorch_transformers:PyTT_WordPiecer",
                "pytt_tok2vec = spacy_pytorch_transformers:PyTT_TokenVectorEncoder",
                "pytt_textcat = spacy_pytorch_transformers:PyTT_TextCategorizer",
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
