"""Playwright wrapper exposed as discrete tools."""
from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import sync_playwright, Page, Browser


@dataclass
class PageState:
    url: str
    title: str
    text: str
    screenshot_b64: str


class BrowserDriver:
    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def start(self) -> None:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page()

    def goto(self, url: str) -> PageState:
        assert self._page
        self._page.goto(url, wait_until="domcontentloaded")
        return self.snapshot()

    def click(self, selector: str) -> PageState:
        assert self._page
        self._page.click(selector)
        return self.snapshot()

    def type(self, selector: str, text: str) -> PageState:
        assert self._page
        self._page.fill(selector, text)
        return self.snapshot()

    def snapshot(self) -> PageState:
        assert self._page
        import base64
        png = self._page.screenshot(type="png")
        return PageState(
            self._page.url,
            self._page.title(),
            self._page.inner_text("body"),
            base64.b64encode(png).decode(),
        )

    def close(self) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
