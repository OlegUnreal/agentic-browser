"""Offline demo: stub LLM + no real browser (skips Playwright)."""
from __future__ import annotations

from .agent import run


def stub_llm(goal: str, url: str, elements: list, history: list, **kwargs):
    if "Elements" in str(elements):
        return {"name": "click", "args": {"selector": "#login"}}
    if history:
        return {"name": "done", "args": {}}
    return {"name": "click", "args": {"selector": "#login"}}


def main() -> None:
    steps = run("Log in", "https://example.com", stub_llm)
    for s in steps:
        print(f"step {s.n}: {s.action} -> {s.result}")


if __name__ == "__main__":
    main()
