from spacy import Language
from spacy.tests.util import assert_docs_equal
from spacy.tokens import Doc

from spacy_transformers import TransformerData
import srsly
from numpy.testing import assert_array_equal


def test_serialize_transformer_data():
    data = {"x": TransformerData.empty()}
    bytes_data = srsly.msgpack_dumps(data)
    new_data = srsly.msgpack_loads(bytes_data)
    assert isinstance(new_data["x"], TransformerData)

    nlp = Language()
    trf = nlp.add_pipe("transformer", config={"model": {"name": "distilbert-base-uncased", "transformer_config": {"output_attentions": True}}})
    nlp.initialize()
    doc = nlp("This is a test.")
    b = doc.to_bytes()
    reloaded_doc = Doc(nlp.vocab)
    reloaded_doc.from_bytes(b)
    assert_docs_equal(doc, reloaded_doc)
    assert_array_equal(doc._.trf_data.tensors, reloaded_doc._.trf_data.tensors)
    for key in doc._.trf_data.model_output:
        assert_array_equal(doc._.trf_data.model_output[key], reloaded_doc._.trf_data.model_output[key])


def test_transformer_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer")
    trf2.from_bytes(trf_bytes)


def test_transformer_model_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    nlp.initialize()
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer")
    nlp2.initialize()
    trf2.from_bytes(trf_bytes)
