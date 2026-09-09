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


def _execute(driver: BrowserDriver, decision: dict, guard: Guard) -> PageState:
    name = decision.get("name", "").lower()
    args = decision.get("args") or {}
    if name == "goto":
        url = args.get("url", "")
        if guard.allow_url(url):
            return driver.goto(url)
        return PageState(url, "blocked", "domain not allowed", "")
    if name == "click":
        return driver.click(args.get("selector", ""))
    if name == "type":
        return driver.type(args.get("selector", ""), args.get("text", ""))
    if name == "done":
        return driver.snapshot()
    return driver.snapshot()


def run(goal: str, start_url: str, llm, max_steps: int = 10,
        allowed_domains: set[str] | None = None) -> list[Step]:
    driver = BrowserDriver()
    guard = Guard(allowed_domains or {"example.com"})
    history: list[Step] = []
    try:
        driver.start()
        if not guard.allow_url(start_url):
            return history
        state = driver.goto(start_url)
        for n in range(1, max_steps + 1):
            elements = describe(state, llm)
            try:
                decision = llm(goal, state.url, elements, history)
            except Exception as exc:  # noqa: BLE001
                history.append(Step(n, "error", str(exc)))
                continue
            if not isinstance(decision, dict):
                decision = {"name": str(decision), "args": {}}
            name = str(decision.get("name", "")).lower()
            if name == "done":
                history.append(Step(n, "DONE", "goal reached"))
                break
            if not guard.allow_action():
                history.append(Step(n, name or "action", "rate-limited"))
                continue
            try:
                state = _execute(driver, decision, guard)
            except Exception as exc:  # noqa: BLE001
                state = PageState(state.url, state.title, f"action failed: {exc}", state.screenshot_b64)
            history.append(Step(n, name or "action", state.url))
    finally:
        driver.close()
    return history
