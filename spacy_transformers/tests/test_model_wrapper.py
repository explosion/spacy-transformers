import pytest
import spacy
from thinc.api import Model
from ..layers import TransformerModel
from ..data_classes import FullTransformerBatch
from ..span_getters import get_doc_spans


MODEL_NAMES = ["distilbert-base-uncased", "gpt2", "xlnet-base-cased"]


@pytest.fixture
def nlp():
    return spacy.blank("en")


@pytest.fixture
def docs(nlp):
    texts = ["the cat sat on the mat.", "hello world."]
    return [nlp(text) for text in texts]


@pytest.fixture(scope="module", params=MODEL_NAMES)
def name(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def output_attentions(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def output_hidden_states(request):
    return request.param


@pytest.fixture(scope="module")
def trf_model(name, output_attentions, output_hidden_states):
    if name == "gpt2":
        model = TransformerModel(
            name,
            get_doc_spans,
            {"use_fast": True, "pad_token": "<|endoftext|>"},
            {
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
            },
        )
    else:
        model = TransformerModel(
            name,
            get_doc_spans,
            {"use_fast": True},
            {
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
            },
        )
    model.initialize()
    return model


def test_model_init(name, trf_model):
    assert isinstance(trf_model, Model)


def test_model_predict(docs, trf_model):
    outputs = trf_model.predict(docs)
    shape = outputs.model_output.last_hidden_state.shape
    if trf_model.transformer.config.output_attentions is True:
        assert outputs.model_output.attentions is not None
        assert all([t.shape[0] == shape[0] for t in outputs.model_output.attentions])
    else:
        assert outputs.model_output.attentions is None
    if trf_model.transformer.config.output_hidden_states is True:
        assert outputs.model_output.hidden_states is not None
        assert all([t.shape[0] == shape[0] for t in outputs.model_output.hidden_states])
    else:
        assert outputs.model_output.hidden_states is None
    assert isinstance(outputs, FullTransformerBatch)
