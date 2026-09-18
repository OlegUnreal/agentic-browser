"""Policy layer: block disallowed domains and cap action rate.

The guard is the only place where a model-authored action can be refused.
Every budget is a rolling 60-second window, and every tool name has to be one
the driver actually implements — a hallucinated verb is refused rather than
silently degraded into a snapshot.
"""
from __future__ import annotations

import time
from collections import Counter
from urllib.parse import urlparse

#: Tool names `_execute` knows how to run. `done` is terminal and handled by the
#: loop, but it is still part of the advertised schema.
KNOWN_TOOLS = frozenset({"goto", "click", "type", "scroll", "done"})

#: Actions that only move the viewport, not the page state.
CHEAP_ACTIONS = frozenset({"scroll"})


class Guard:
    def __init__(
        self,
        allowed_domains: set[str],
        max_actions_per_min: int = 30,
        max_scrolls_per_min: int = 12,
        clock=time.time,
    ):
        self.allowed = set(allowed_domains or set())
        self.max_per_min = int(max_actions_per_min)
        self.max_scrolls_per_min = int(max_scrolls_per_min)
        self._clock = clock
        self._timestamps: list[float] = []
        self._scrolls: list[float] = []
        self.denials: Counter[str] = Counter()

    # -- domain policy -----------------------------------------------------
    def allow_url(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        allowed = any(host == d or host.endswith("." + d) for d in self.allowed)
        if not allowed:
            self.denials["url"] += 1
        return allowed

    # -- tool policy -------------------------------------------------------
    def allow_tool(self, name: str) -> bool:
        """Refuse tool names the driver cannot execute."""
        known = str(name or "").lower() in KNOWN_TOOLS
        if not known:
            self.denials["unknown_tool"] += 1
        return known

    # -- rate policy -------------------------------------------------------
    def _prune(self, now: float) -> None:
        horizon = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > horizon]
        self._scrolls = [t for t in self._scrolls if t > horizon]

    def allow_action(self, kind: str = "action") -> bool:
        """Consume one action budget for `kind`.

        Backwards compatible: called with no arguments it behaves like the
        original global cap. `scroll` additionally consumes the separate,
        smaller scroll budget so view-spamming cannot eat the whole minute.
        """
        now = self._clock()
        self._prune(now)
        name = str(kind or "").lower()
        if name in CHEAP_ACTIONS and len(self._scrolls) >= self.max_scrolls_per_min:
            self.denials["scroll_rate"] += 1
            return False
        if len(self._timestamps) >= self.max_per_min:
            self.denials["rate"] += 1
            return False
        self._timestamps.append(now)
        if name in CHEAP_ACTIONS:
            self._scrolls.append(now)
        return True

    def remaining(self) -> dict[str, int]:
        """Budget left in the current window — surfaced in the trace."""
        now = self._clock()
        self._prune(now)
        return {
            "actions": max(0, self.max_per_min - len(self._timestamps)),
            "scrolls": max(0, self.max_scrolls_per_min - len(self._scrolls)),
        }
