from agentic.agent import run


def stub(prompt, history=None):
    return "DONE"


def test_terminates():
    # Without a real browser this just checks the stub path doesn't crash.
    steps = run("goal", "https://example.com", stub)
    assert isinstance(steps, list)
