"""Offline demo: stub LLM + no real browser (skips Playwright)."""
from __future__ import annotations

from .agent import run


def stub_llm(prompt: str, history=None):
    if "Elements" in prompt:
        return [{"selector": "#login", "label": "Login", "type": "button"}]
    if "Choose one action" in prompt:
        return "DONE"
    return []


def main() -> None:
    steps = run("Log in", "https://example.com", stub_llm)
    for s in steps:
        print(f"step {s.n}: {s.action} -> {s.result}")


if __name__ == "__main__":
    main()
