import pytest
from spacy.lang.en import English

from ..span_getters import configure_strided_spans, configure_get_sent_spans
from ..util import registry


@pytest.mark.parametrize(
    "window,stride,docs,result",
    [
        (4, 3, ["0", "1234", "56789a"], [["0"], ["1234"], ["5678", "89a"]]),
        (4, 4, ["0", "1234", "56789a"], [["0"], ["1234"], ["5678", "9a"]]),
        (4, 2, ["0", "1234", "56789a"], [["0"], ["1234"], ["5678", "789a"]]),
    ],
)
def test_get_strided_spans(window, stride, docs, result):
    get_strided = configure_strided_spans(window, stride)
    spans = get_strided(docs)
    assert spans == result


def test_get_sent_spans():
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp("One. One more. Three sentences in total.")
    assert len(list(doc.sents)) == 3
    get_sent_spans = configure_get_sent_spans()
    spans = get_sent_spans([doc])[0]
    assert len(spans) == 3
    assert spans[0].text == "One."
    assert spans[1].text == "One more."
    assert spans[2].text == "Three sentences in total."


def test_get_custom_spans():
    def configure_custom_sent_spans(max_length: int):
        def get_custom_sent_spans(docs):
            spans = []
            for doc in docs:
                spans.append([])
                for sent in doc.sents:
                    start = 0
                    end = max_length
                    while end <= len(sent):
                        spans[-1].append(sent[start:end])
                        start += max_length
                        end += max_length
                    if start < len(sent):
                        spans[-1].append(sent[start:len(sent)])
            return spans

        return get_custom_sent_spans

    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(
        "One. And one more. So that makes three sentences and this one is a bit longer."
    )
    assert len(list(doc.sents)) == 3
    get_sent_spans = configure_custom_sent_spans(max_length=4)
    spans = get_sent_spans([doc])[0]
    assert len(spans) == 6
    assert spans[0].text == "One."
    assert spans[1].text == "And one more."
    assert spans[2].text == "So that makes three"
    assert spans[3].text == "sentences and this one"
    assert spans[4].text == "is a bit longer"
    assert spans[5].text == "."
