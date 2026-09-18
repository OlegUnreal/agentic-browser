"""Playwright wrapper exposed as discrete tools.

Playwright is imported *lazily*: only :meth:`BrowserDriver.start` needs it.
That keeps `agentic.agent`, the perception layer and the whole offline test
suite importable on a machine with no browser binary installed, which is the
point of the fakes in `tests/conftest.py`.

`PageState.elements` carries a normalised interactive-element list extracted
from the live accessibility/DOM tree (see `snapshot.parse_dom_json`), which is
what the retrieval layer embeds and ranks.
"""
from __future__ import annotations

import base64
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import SETTINGS
from .logging_config import get_logger
from .snapshot import parse_dom_json, structure_hash

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from playwright.sync_api import Browser, Page, SyncPlaywright

log = get_logger(__name__)

#: Default vertical scroll delta for the `scroll` tool, in CSS pixels.
SCROLL_STEP = 500

_TIMEOUT_MS = 15000
_ACTION_TIMEOUT_MS = 10000


class BrowserUnavailable(RuntimeError):
    """Raised when a real browser is requested but Playwright is not installed."""


def _import_playwright():
    """Import `playwright.sync_api`, falling back to an installed fake module.

    The test suite injects `playwright.sync_api` into `sys.modules` with a
    recording double, which lets `BrowserDriver` be exercised end-to-end
    (headed flag, scroll wheel, tool dispatch) with no browser present.
    """
    module = sys.modules.get("playwright.sync_api")
    if module is not None:
        return getattr(module, "sync_playwright")
    try:  # pragma: no cover - exercised only when Playwright is installed
        from playwright.sync_api import sync_playwright as factory
    except ImportError as exc:  # pragma: no cover
        raise BrowserUnavailable(
            "playwright is not installed; run `pip install playwright && playwright install chromium` "
            "to drive a real browser, or use `python -m agentic.demo` for the scripted, browser-free "
            "walkthrough of the same agent loop"
        ) from exc
    return factory


#: JS evaluated by :meth:`BrowserDriver.extract_elements`. Walks the real
#: accessibility-relevant nodes, keeps only visible interactive ones, builds a
#: best-effort unique CSS selector (id > name > aria-label > nth-of-type path),
#: and reports geometry so downstream features can reason about position.
_DOM_EXTRACT_JS = r"""
(args) => {
  const limit = args.limit;
  const interactive = 'a,button,input,select,textarea,[role],[onclick],[tabindex],summary,label';
  function cssPath(el) {
    if (el.id) return '#' + CSS.escape(el.id);
    const name = el.getAttribute && el.getAttribute('name');
    if (name && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA'))
      return el.tagName.toLowerCase() + '[name="' + name + '"]';
    const label = el.getAttribute && el.getAttribute('aria-label');
    if (label) {
      const uniq = document.querySelectorAll('[aria-label="' + label + '"]').length === 1;
      if (uniq) return el.tagName.toLowerCase() + '[aria-label="' + label + '"]';
    }
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 6) {
      let part = node.tagName.toLowerCase();
      const parent = node.parentElement;
      if (parent) {
        const same = Array.from(parent.children).filter((c) => c.tagName === node.tagName);
        if (same.length > 1) part += ':nth-of-type(' + (same.indexOf(node) + 1) + ')';
      }
      parts.unshift(part);
      if (node.id) { parts[parts.length - 1] = '#' + CSS.escape(node.id); break; }
      node = parent;
    }
    return parts.join(' > ');
  }
  function textOf(el) {
    const aria = el.getAttribute && el.getAttribute('aria-label');
    const value = el.value && ['button','submit','reset'].includes((el.type||'').toLowerCase()) ? el.value : '';
    let txt = aria || el.innerText || value || el.textContent ||
              (el.getAttribute && (el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt'))) || '';
    return String(txt).replace(/\s+/g, ' ').trim().slice(0, 160);
  }
  const vh = window.innerHeight || 0;
  const vw = window.innerWidth || 0;
  const seen = new Set();
  const out = [];
  for (const el of Array.from(document.querySelectorAll(interactive))) {
    if (out.length >= limit) break;
    const style = window.getComputedStyle(el);
    const hidden = style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0;
    if (hidden || el.disabled) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const selector = cssPath(el);
    if (seen.has(selector)) continue;
    seen.add(selector);
    const role = el.getAttribute && el.getAttribute('role')
      ? el.getAttribute('role')
      : ({a:'link', button:'button', select:'combobox', textarea:'textbox'}[el.tagName.toLowerCase()]
         || (el.tagName === 'INPUT' ? ({checkbox:'checkbox', radio:'radio', submit:'button', search:'searchbox'}[(el.type||'').toLowerCase()] || 'textbox') : 'generic'));
    out.push({
      selector,
      role,
      tag: el.tagName.toLowerCase(),
      text: textOf(el),
      attrs: {
        id: el.id || '', name: el.getAttribute('name') || '', type: el.getAttribute('type') || '',
        placeholder: el.getAttribute('placeholder') || '', 'aria-label': el.getAttribute('aria-label') || '',
        title: el.getAttribute('title') || '', alt: el.getAttribute('alt') || '', href: el.getAttribute('href') || '',
      },
      rect: {x: Math.round(r.x), y: Math.round(r.y + window.scrollY), width: Math.round(r.width), height: Math.round(r.height)},
      visible: true,
      inViewport: r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw,
    });
  }
  return JSON.stringify({url: location.href, title: document.title, elements: out});
}
"""


@dataclass
class PageState:
    url: str
    title: str
    text: str
    screenshot_b64: str
    #: Normalised interactive elements from the accessibility/DOM tree.
    elements: list[dict] = field(default_factory=list)
    #: Raw viewport height, used for in-viewport feature extraction.
    viewport_height: int = 900

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "elements": self.elements,
        }

    def structure_hash(self) -> str:
        """Stable identity of this state for episodic-memory loop detection."""
        return structure_hash(self.url, self.elements, self.title)


class BrowserDriver:
    """Thin, explicit tool surface over one Playwright page."""

    def __init__(self, headed: bool | None = None, slow_mo_ms: int = 0) -> None:
        # `None` means "decide from config", which reads AGENT_HEADED.
        self.headed = SETTINGS.headed if headed is None else bool(headed)
        self.slow_mo_ms = int(slow_mo_ms)
        self._pw: SyncPlaywright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        factory = _import_playwright()
        self._pw = factory().start()
        self._browser = self._pw.chromium.launch(
            headless=not self.headed,
            slow_mo=self.slow_mo_ms,
        )
        self._page = self._browser.new_page()
        log.info(
            "browser_started",
            extra={"headed": self.headed, "chromium": getattr(self._pw, "chromium_version", lambda: "?")()},
        )

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()
            self._browser = None
            self._pw = None
            self._page = None

    def __enter__(self) -> "BrowserDriver":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- tools -------------------------------------------------------------
    def goto(self, url: str) -> PageState:
        assert self._page, "driver not started"
        self._page.goto(url, wait_until="domcontentloaded", timeout=_TIMEOUT_MS)
        return self.snapshot()

    def click(self, selector: str) -> PageState:
        assert self._page, "driver not started"
        self._page.click(selector, timeout=_ACTION_TIMEOUT_MS)
        return self.snapshot()

    def type(self, selector: str, text: str) -> PageState:
        assert self._page, "driver not started"
        self._page.fill(selector, text, timeout=_ACTION_TIMEOUT_MS)
        return self.snapshot()

    def scroll(self, selector: str = "", amount: int = SCROLL_STEP) -> PageState:
        """Scroll the page, or bring `selector` into view when given.

        `amount` is a signed vertical delta in CSS pixels: positive goes down
        the document, negative goes back up. This is the tool that lets the
        agent reach content the snapshot's viewport window cut off.
        """
        assert self._page, "driver not started"
        delta = int(amount or 0)
        if selector:
            self._page.eval_on_selector(
                selector, "(el) => el.scrollIntoView({block: 'center'})", timeout=_ACTION_TIMEOUT_MS
            )
        else:
            self._page.mouse.wheel(0, delta)
        self._page.wait_for_timeout(60)  # let lazy loaders settle
        return self.snapshot()

    def snapshot(self) -> PageState:
        assert self._page, "driver not started"
        png = self._page.screenshot(type="png", full_page=False)
        return PageState(
            url=self._page.url,
            title=self._page.title(),
            text=self._page.inner_text("body"),
            screenshot_b64=base64.b64encode(png).decode("ascii"),
            elements=self.extract_elements(),
            viewport_height=int(self._page.evaluate("() => window.innerHeight") or 900),
        )

    def extract_elements(self, limit: int = 400) -> list[dict]:
        """Interactive elements from the live page, normalised for perception."""
        assert self._page, "driver not started"
        try:
            raw = self._page.evaluate(_DOM_EXTRACT_JS, {"limit": int(limit)})
        except Exception as exc:  # noqa: BLE001 - a partial page must not kill the loop
            log.warning("dom_extract_failed", extra={"error": type(exc).__name__})
            return []
        return parse_dom_json(raw if isinstance(raw, str) else None, limit=limit)

    # -- non-tool helpers used by the agent for recovery -------------------
    def scroll_to_top(self) -> PageState:
        assert self._page, "driver not started"
        self._page.evaluate("() => window.scrollTo(0, 0)")
        return self.snapshot()

    @staticmethod
    def from_env() -> "BrowserDriver":
        """Build a driver honouring `AGENT_HEADED`."""
        return BrowserDriver()
