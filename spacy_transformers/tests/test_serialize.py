from spacy_transformers import TransformerData
import srsly


def test_serialize_transformer_data():
    data = {"x": TransformerData.empty()}
    bytes_data = srsly.msgpack_dumps(data)
    new_data = srsly.msgpack_loads(bytes_data)
    assert isinstance(new_data["x"], TransformerData)
