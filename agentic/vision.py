"""Vision layer: describe the page for the agent."""
from __future__ import annotations

from .browser import PageState

VISION_PROMPT = """Describe the interactive elements on this page.
Return JSON list of {{"selector": str, "label": str, "type": "button|link|input"}}.
Page text: {text}
"""


def describe(state: PageState, llm) -> list[dict]:
    return llm(VISION_PROMPT.format(text=state.text[:4000]))
