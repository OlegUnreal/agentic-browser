"""Real LLM brain for the agentic browser.

Replaces the stub decision parser with structured tool calls via OpenAI,
with retries and safe fallbacks.
"""
from __future__ import annotations

import json
from typing import Callable

from .config import SETTINGS
from .logging_config import get_logger

log = get_logger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "goto",
            "description": "Navigate to a URL",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element by CSS selector",
            "parameters": {"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type text into an element",
            "parameters": {"type": "object", "properties": {"selector": {"type": "string"}, "text": {"type": "string"}}, "required": ["selector", "text"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Goal reached, stop",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _parse_tool_call(msg) -> dict:
    if not getattr(msg, "tool_calls", None):
        return {"name": "done", "args": {}}
    tc = msg.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": tc.function.name, "args": args}


def make_llm(model: str | None = None) -> Callable:
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError

    m = model or SETTINGS.model
    client = OpenAI(api_key=SETTINGS.openai_api_key, timeout=SETTINGS.timeout)

    def decide(goal: str, url: str, elements, history: list) -> dict:
        hist = "\n".join(f"- {h.action}: {h.result}" for h in history[-5:])
        elem_txt = json.dumps(elements) if not isinstance(elements, str) else elements
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "You control a browser. Pick exactly one tool call."},
                        {"role": "user", "content": f"Goal: {goal}\nURL: {url}\nElements: {elem_txt}\nHistory:\n{hist}"},
                    ],
                    tools=TOOLS,
                    tool_choice="required",
                    timeout=SETTINGS.timeout,
                )
                return _parse_tool_call(resp.choices[0].message)
            except (APIError, RateLimitError, APITimeoutError) as exc:
                last = exc
                log.warning("llm_retry", extra={"attempt": attempt + 1, "error": type(exc).__name__})
                continue
        log.error("llm_exhausted", extra={"error": type(last).__name__ if last else None})
        return {"name": "done", "args": {}}

    return decide
