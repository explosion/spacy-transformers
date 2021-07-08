from spacy import Language

from spacy_transformers import TransformerData
import srsly


def test_serialize_transformer_data():
    data = {"x": TransformerData.empty()}
    bytes_data = srsly.msgpack_dumps(data)
    new_data = srsly.msgpack_loads(bytes_data)
    assert isinstance(new_data["x"], TransformerData)


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
