from agentic.vision import _parse_elements, describe
from agentic.browser import PageState


def test_parse_list():
    assert _parse_elements([{"selector": "#a", "label": "A", "type": "button"}])[0]["selector"] == "#a"


def test_parse_fenced():
    raw = '```json\n[{"selector": "#a", "label": "A", "type": "button"}]\n```'
    assert _parse_elements(raw)[0]["label"] == "A"


def test_parse_garbage():
    assert _parse_elements("nope") == []


def test_describe_wraps():
    state = PageState("u", "t", "hello", "")
    out = describe(state, lambda p: [{"selector": "#x", "label": "X", "type": "link"}])
    assert out[0]["selector"] == "#x"
