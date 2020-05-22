import pytest
from ..span_getters import configure_strided_spans


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
