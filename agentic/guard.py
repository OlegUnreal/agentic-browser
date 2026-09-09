"""Policy layer: block disallowed domains and cap action rate."""
from __future__ import annotations

import time
from urllib.parse import urlparse


class Guard:
    def __init__(self, allowed_domains: set[str], max_actions_per_min: int = 30):
        self.allowed = allowed_domains
        self.max_per_min = max_actions_per_min
        self._timestamps: list[float] = []

    def allow_url(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in self.allowed)

    def allow_action(self) -> bool:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_per_min:
            return False
        self._timestamps.append(now)
        return True
