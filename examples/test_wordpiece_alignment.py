#!/usr/bin/env python
import plac
from spacy_pytorch_transformers import TransformersWordPiecer
from spacy.util import get_lang_class
import thinc.extra.datasets
from wasabi import Printer, color
import difflib
import sys
import tqdm

msg = Printer()


@plac.annotations(
    name=("Pretrained model name, e.g. 'bert-base-uncased'", "positional", None, str),
    n_texts=("Number of texts to load (0 for all)", "option", "n", int),
    lang=("spaCy language to use for tokenization", "option", "l", str),
    skip=("Don't stop processing on errors, only print", "flag", "s", bool),
    retry=("Enable retrying alignment by aggressive replacement", "flag", "r", bool),
    force=("Enable forced alignment via tokens", "flag", "f", bool),
)
def main(
    name="bert-base-uncased",
    n_texts=1000,
    lang="en",
    skip=False,
    retry=False,
    force=False,
):
    """Test the wordpiecer on a large dataset to find misalignments. If both the
    retry and force flag are set (which is the default runtime configuration),
    this script should always pass.

    * retry: If alignment fails after cleaning and normalizing both sets of
        tokens, try again with a more aggressive strategy that strips out all
        characters that are not uppercase/lowercase letters.
    * force: If alignment still fails, run the word-piece tokenizer on the
        individual spaCy tokens, so that alignment is trivial. This should
        always work.
    """
    cfg = {"retry_alignment": retry, "force_alignment": force}
    nlp = get_lang_class(lang)()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    wp = TransformersWordPiecer.from_pretrained(nlp.vocab, trf_name=name, **cfg)
    msg.good(f"Loaded WordPiecer for model '{name}'")
    with msg.loading("Loading IMDB data..."):
        data, _ = thinc.extra.datasets.imdb(limit=n_texts)
    texts, _ = zip(*data)
    msg.good(f"Using {len(texts)} texts from IMDB data")
    msg.info("Processing texts...")
    sent_counts = 0
    for doc in tqdm.tqdm(nlp.pipe(texts), total=len(texts)):
        try:
            doc = wp(doc)
            sent_counts += len(list(doc.sents))
        except AssertionError as e:
            if len(e.args) and isinstance(e.args[0], tuple):  # Misaligned error
                a, b = e.args[0]
                msg.fail("Misaligned tokens")
                print(diff_strings(a, b))
                if not skip:
                    sys.exit(1)
            elif len(e.args):
                msg.fail(f"Error: {e.args[0]}", exits=None if skip else 1)
            else:
                if skip:
                    print(e)
                else:
                    raise e
    msg.good(f"Processed {len(texts)} documents ({sent_counts} sentences)")


def diff_strings(a, b):
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            output.append(a[a0:a1])
        elif opcode == "insert":
            output.append(color(b[b0:b1], fg=16, bg="green"))
        elif opcode == "delete":
            output.append(color(a[a0:a1], fg=16, bg="red"))
        elif opcode == "replace":
            output.append(color(b[b0:b1], fg=16, bg="green"))
            output.append(color(a[a0:a1], fg=16, bg="red"))
    return "".join(output)


if __name__ == "__main__":
    plac.call(main)
