"""Offline test doubles.

Nothing in this suite touches a browser or a network. `FakeDriver` records the
tool calls the agent makes; `ScriptedLLM` replays a canned decision trace and
raises if the agent asks for more steps than the script covers (so a loop that
fails to terminate fails loudly instead of hanging); `fake_playwright` installs a
module-level double for `playwright.sync_api` so `BrowserDriver` itself can be
driven end-to-end — headed flag, scroll wheel, snapshots — with no browser.
"""
from __future__ import annotations

import sys
import types
from typing import Any, Iterable, Sequence

import pytest

from agentic.browser import PageState
from agentic.perception.feedback import FeedbackStore
from agentic.perception.memory import EpisodicMemory
from agentic.perception.selectors import SelectionContext
from agentic.snapshot import coerce_elements

# --------------------------------------------------------------------------
# synthetic page
# --------------------------------------------------------------------------

CHECKOUT_ELEMENTS: list[dict] = [
    {"selector": "#search", "label": "Search products", "role": "searchbox", "index": 0,
     "attrs": {"placeholder": "Search products"}, "rect": {"x": 10, "y": 20, "width": 200, "height": 30},
     "visible": True, "in_viewport": True},
    {"selector": "#cart-link", "label": "Shopping cart (3 items)", "role": "link", "index": 1,
     "attrs": {"href": "/cart"}, "rect": {"x": 900, "y": 20, "width": 80, "height": 30},
     "visible": True, "in_viewport": True},
    {"selector": "body > div:nth-of-type(2) > ul > li:nth-of-type(1) > button", "label": "Add to cart",
     "role": "button", "index": 2, "attrs": {}, "rect": {"x": 40, "y": 300, "width": 120, "height": 32},
     "visible": True, "in_viewport": True},
    {"selector": "body > div:nth-of-type(2) > ul > li:nth-of-type(2) > button", "label": "Add to cart",
     "role": "button", "index": 3, "attrs": {}, "rect": {"x": 40, "y": 420, "width": 120, "height": 32},
     "visible": True, "in_viewport": True},
    {"selector": "body > div:nth-of-type(2) > ul > li:nth-of-type(3) > button", "label": "Add to cart",
     "role": "button", "index": 4, "attrs": {}, "rect": {"x": 40, "y": 540, "width": 120, "height": 32},
     "visible": True, "in_viewport": False},
    {"selector": "#newsletter-submit", "label": "Subscribe to our weekly newsletter", "role": "button",
     "index": 5, "attrs": {}, "rect": {"x": 40, "y": 1400, "width": 160, "height": 32},
     "visible": True, "in_viewport": False},
    {"selector": "#coupon-code", "label": "Coupon code", "role": "textbox", "index": 6,
     "attrs": {"placeholder": "Enter coupon code"}, "rect": {"x": 40, "y": 1500, "width": 220, "height": 32},
     "visible": True, "in_viewport": False},
    {"selector": "#apply-coupon", "label": "Apply coupon", "role": "button", "index": 7, "attrs": {},
     "rect": {"x": 280, "y": 1500, "width": 110, "height": 32}, "visible": True, "in_viewport": False},
    {"selector": "#delivery-address-edit", "label": "Edit delivery address", "role": "button", "index": 8,
     "attrs": {}, "rect": {"x": 40, "y": 1620, "width": 150, "height": 32}, "visible": True, "in_viewport": False},
    {"selector": "#promo-banner", "label": "Summer sale banner", "role": "generic", "index": 9, "attrs": {},
     "rect": {"x": 0, "y": 0, "width": 0, "height": 0}, "visible": False, "in_viewport": False},
    {"selector": "#footer-terms", "label": "Terms and conditions", "role": "link", "index": 10,
     "attrs": {"href": "/terms"}, "rect": {"x": 40, "y": 3000, "width": 120, "height": 20},
     "visible": True, "in_viewport": False},
    {"selector": "#pay-now", "label": "Pay now", "role": "button", "index": 11, "attrs": {},
     "rect": {"x": 40, "y": 1800, "width": 120, "height": 36}, "visible": True, "in_viewport": False},
]


@pytest.fixture
def checkout_elements() -> list[dict]:
    """A synthetic checkout page: near-duplicate 'Add to cart' buttons plus the
    one widget in the coupon area the retrieval has to disambiguate."""
    return coerce_elements(CHECKOUT_ELEMENTS)


@pytest.fixture
def checkout_state(checkout_elements) -> PageState:
    return PageState(
        url="https://shop.example.com/checkout",
        title="Checkout",
        text="Your cart Total Apply coupon",
        screenshot_b64="",
        elements=checkout_elements,
    )


def make_state(url: str = "https://shop.example.com/", elements: Sequence[dict] | None = None, title: str = "t") -> PageState:
    return PageState(url=url, title=title, text="body", screenshot_b64="", elements=list(elements or []))


# --------------------------------------------------------------------------
# fake driver
# --------------------------------------------------------------------------


class FakeDriver:
    """Records tool calls and replays scripted page states. No Playwright."""

    def __init__(self, states: Iterable[PageState] | None = None, raise_on: dict[str, Exception] | None = None):
        self.calls: list[tuple] = []
        self.started = 0
        self.closed = 0
        self.headed: bool | None = None
        self._states = list(states or [])
        self._raise_on = raise_on or {}

    def _next(self, default_url: str = "https://example.com/") -> PageState:
        if self._states:
            return self._states.pop(0)
        return PageState(url=default_url, title="fake", text="body", screenshot_b64="", elements=[])

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    def goto(self, url: str) -> PageState:
        self.calls.append(("goto", url))
        self._maybe_raise("goto")
        return self._next(url)

    def click(self, selector: str) -> PageState:
        self.calls.append(("click", selector))
        self._maybe_raise("click")
        return self._next()

    def type(self, selector: str, text: str) -> PageState:
        self.calls.append(("type", selector, text))
        self._maybe_raise("type")
        return self._next()

    def scroll(self, selector: str = "", amount: int = 500) -> PageState:
        self.calls.append(("scroll", selector, amount))
        self._maybe_raise("scroll")
        return self._next()

    def snapshot(self) -> PageState:
        self.calls.append(("snapshot",))
        return self._next()

    def _maybe_raise(self, action: str) -> None:
        exc = self._raise_on.get(action)
        if exc is not None:
            raise exc

    # convenience for assertions
    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture
def driver() -> FakeDriver:
    return FakeDriver()


@pytest.fixture
def fake_driver_factory():
    return lambda states=None, raise_on=None: FakeDriver(states=states, raise_on=raise_on)


# --------------------------------------------------------------------------
# scripted offline LLM
# --------------------------------------------------------------------------


class ScriptedLLM:
    """Offline stand-in for `make_llm()`: replays decisions, records prompts.

    Raises `AssertionError` when the script runs out, which turns a loop that
    fails to terminate into a failed test rather than an infinite one.
    """

    def __init__(self, decisions: Sequence[Any] | None = None, default: Any | None = None):
        self.decisions = list(decisions or [])
        self.default = default
        self.prompts: list[dict] = []
        self.calls = 0

    def __call__(self, goal: str, url: str, elements, history, context: dict | None = None) -> dict:
        self.calls += 1
        self.prompts.append(
            {
                "goal": goal,
                "url": url,
                "elements": list(elements or []),
                "history": list(history or []),
                "context": context,
            }
        )
        if self.decisions:
            decision = self.decisions.pop(0)
        elif self.default is not None:
            decision = self.default
        else:
            raise AssertionError(f"ScriptedLLM exhausted after {self.calls} calls (element list had no match?)")
        if isinstance(decision, Exception):
            raise decision
        return decision

    @property
    def seen_elements(self) -> list[list]:
        return [p["elements"] for p in self.prompts]


@pytest.fixture
def script_llm():
    """Factory: `llm = script_llm([{"name": "click", ...}])`."""
    return lambda decisions, default=None: ScriptedLLM(decisions, default=default)


# --------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------


@pytest.fixture
def memory() -> EpisodicMemory:
    store = EpisodicMemory(":memory:")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def feedback() -> FeedbackStore:
    store = FeedbackStore(":memory:")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def context() -> SelectionContext:
    return SelectionContext(goal="apply the coupon code", action="click")


# --------------------------------------------------------------------------
# playwright double
# --------------------------------------------------------------------------


class FakePage:
    def __init__(self, recording: dict, url: str = "https://example.com/"):
        self._rec = recording
        self.url = url
        self.mouse = types.SimpleNamespace(wheel=lambda dx, dy: recording["wheels"].append((dx, dy)))
        self._scroll_top = False

    def goto(self, url, **kwargs):
        self.url = url
        self._rec["goto"].append((url, kwargs))

    def click(self, selector, **kwargs):
        self._rec["click"].append((selector, kwargs))

    def fill(self, selector, text, **kwargs):
        self._rec["fill"].append((selector, text, kwargs))

    def eval_on_selector(self, selector, script, **kwargs):
        self._rec["eval_on_selector"].append((selector, script[:24], kwargs))

    def evaluate(self, script, arg=None):
        self._rec["evaluate"].append((script[:32], arg))
        if "innerHeight" in script:
            return 800
        return '{"elements": []}' if "querySelectorAll" in script else None

    def wait_for_timeout(self, ms):
        self._rec["waits"].append(ms)

    def screenshot(self, **kwargs):
        self._rec["screenshot"].append(kwargs)
        return b"\x89PNG\r\n\x1a\nfake"

    def title(self):
        return "Fake page"

    def inner_text(self, selector):
        return "fake body text"


class FakeBrowser:
    def __init__(self, recording: dict):
        self._rec = recording

    def new_page(self):
        page = FakePage(self._rec)
        self._rec["pages"].append(page)
        return page

    def close(self):
        self._rec["browser_closed"] = True


class FakeChromium:
    def __init__(self, recording: dict):
        self._rec = recording

    def launch(self, **kwargs):
        self._rec["launch"].append(kwargs)
        return FakeBrowser(self._rec)


class FakePlaywrightManager:
    def __init__(self, recording: dict):
        self._rec = recording

    def start(self):
        self._rec["started"] = True
        return self

    def stop(self):
        self._rec["stopped"] = True

    @property
    def chromium(self):
        return FakeChromium(self._rec)

    def chromium_version(self):
        return "fake"


@pytest.fixture
def fake_playwright(monkeypatch):
    """Install a `playwright.sync_api` double and hand back its call recording."""
    recording: dict[str, Any] = {
        "launch": [], "goto": [], "click": [], "fill": [], "eval_on_selector": [],
        "evaluate": [], "wheels": [], "waits": [], "screenshot": [], "pages": [],
        "started": False, "stopped": False, "browser_closed": False,
    }
    module = types.ModuleType("playwright.sync_api")
    module.sync_playwright = lambda *a, **k: FakePlaywrightManager(recording)
    module.Page = FakePage
    module.Browser = FakeBrowser
    parent = types.ModuleType("playwright")
    parent.sync_api = module
    monkeypatch.setitem(sys.modules, "playwright", parent)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", module)
    return recording
