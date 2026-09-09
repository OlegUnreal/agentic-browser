from agentic.agent import run, _execute
from agentic.browser import PageState
from agentic.guard import Guard
from agentic.vision import _parse_elements, describe


class FakeDriver:
    def __init__(self):
        self.calls = []
    def goto(self, url):
        self.calls.append(("goto", url)); return PageState(url, "t", "body", "")
    def click(self, sel):
        self.calls.append(("click", sel)); return PageState("u", "t", "body", "")
    def type(self, sel, text):
        self.calls.append(("type", sel, text)); return PageState("u", "t", "body", "")
    def snapshot(self):
        return PageState("u", "t", "body", "")


def test_execute_goto_blocked():
    g = Guard({"example.com"})
    d = FakeDriver()
    state = _execute(d, {"name": "goto", "args": {"url": "https://evil.com"}}, g)
    assert state.text == "domain not allowed"
    assert d.calls == []


def test_execute_click():
    g = Guard({"example.com"})
    d = FakeDriver()
    _execute(d, {"name": "click", "args": {"selector": "#x"}}, g)
    assert d.calls[0] == ("click", "#x")


def test_terminates_on_done():
    def stub(goal, url, elements, history):
        return {"name": "done", "args": {}}
    steps = run("goal", "https://example.com", stub)
    assert steps and steps[-1].action == "DONE"


def test_survives_llm_error():
    def boom(goal, url, elements, history):
        raise RuntimeError("llm down")
    steps = run("goal", "https://example.com", boom, max_steps=2)
    assert any(s.action == "error" for s in steps)


def test_parse_elements_tolerates_fences():
    raw = "```json\n[{\"selector\": \"#a\", \"label\": \"Go\", \"type\": \"button\"}]\n```"
    els = _parse_elements(raw)
    assert els and els[0]["selector"] == "#a"


def test_parse_elements_empty():
    assert _parse_elements("") == []
    assert _parse_elements(None) == []


def test_describe_accepts_decision_stub():
    """describe() must tolerate a decision-style stub used by agent tests."""
    state = PageState("https://example.com", "t", "body text", "")
    els = describe(state, lambda g, u, e, h: {"name": "done"})
    assert els == []


def test_describe_survives_crashing_stub():
    state = PageState("https://example.com", "t", "body text", "")
    els = describe(state, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert els == []
