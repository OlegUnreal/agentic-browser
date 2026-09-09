"""Agent loop: observe → choose tool → act, with guard checks."""
from __future__ import annotations

from dataclasses import dataclass

from .browser import BrowserDriver, PageState
from .guard import Guard
from .vision import describe


@dataclass
class Step:
    n: int
    action: str
    result: str


def run(goal: str, start_url: str, llm, max_steps: int = 10) -> list[Step]:
    driver = BrowserDriver()
    guard = Guard(allowed_domains={"example.com"})
    driver.start()
    history: list[Step] = []
    try:
        if not guard.allow_url(start_url):
            return history
        state = driver.goto(start_url)
        for n in range(1, max_steps + 1):
            elements = describe(state, llm)
            decision = llm(
                f"Goal: {goal}\nPage: {state.url}\nElements: {elements}\n"
                f"Choose one action: goto(url) | click(selector) | type(selector, text) | DONE",
                history,
            )
            if decision.upper().startswith("DONE"):
                history.append(Step(n, "DONE", "goal reached"))
                break
            if not guard.allow_action():
                history.append(Step(n, decision, "rate-limited"))
                continue
            state = _execute(driver, decision, guard)
            history.append(Step(n, decision, state.url))
    finally:
        driver.close()
    return history


def _execute(driver: BrowserDriver, decision: str, guard: Guard) -> PageState:
    # Minimal parser: real version uses structured tool calls.
    if decision.startswith("goto"):
        url = decision.split("(", 1)[1].rstrip(")").strip("\"'")
        if guard.allow_url(url):
            return driver.goto(url)
    if decision.startswith("click"):
        sel = decision.split("(", 1)[1].rstrip(")").strip("\"'")
        return driver.click(sel)
    return driver.snapshot()
