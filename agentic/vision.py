"""Vision layer: describe the page for the agent."""
from __future__ import annotations

import json
import re

from .browser import PageState

VISION_PROMPT = """Describe the interactive elements on this page.
Return ONLY a JSON list of objects: {{"selector": str, "label": str, "type": "button|link|input"}}. No markdown.
Page text: {text}
"""

_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_elements(raw) -> list[dict]:
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if not isinstance(raw, str) or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = _JSON_RE.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def describe(state: PageState, llm) -> list[dict]:
    """Return interactive elements. `llm` may be a vision callable
    `(state) -> raw` or a decision callable `(goal, url, elements, history) -> dict`.
    """
    try:
        raw = llm(state)
    except TypeError:
        # Decision-style stub used in tests: (goal, url, elements, history) -> dict.
        raw = llm("goal", state.url, [], [])
    return _parse_elements(raw)
