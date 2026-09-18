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
            "name": "scroll",
            "description": (
                "Scroll the page. With a selector, brings that element into view; "
                "without one, moves the viewport by `amount` pixels (positive = down the document). "
                "Use it when the target is not in the retrieved element list yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Optional CSS selector to bring into view"},
                    "amount": {"type": "integer", "description": "Vertical delta in pixels, default 500"},
                },
                "required": [],
            },
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

#: Names the model is allowed to emit. Kept in sync with `guard.KNOWN_TOOLS`.
TOOL_NAMES = tuple(schema["function"]["name"] for schema in TOOLS)


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


def _serialise_elements(elements) -> str:
    """Render the retrieved element list as compact JSON for the prompt.

    Accepts plain dicts (legacy call sites) as well as the selection objects the
    perception layer returns, which expose `as_dict()` / carry `.element`.
    """
    if isinstance(elements, str):
        return elements
    rows: list[dict] = []
    for item in elements or []:
        if isinstance(item, dict):
            rows.append(item)
            continue
        for attr in ("as_dict", "to_dict"):
            method = getattr(item, attr, None)
            if callable(method):
                payload = method()
                if isinstance(payload, dict):
                    rows.append(payload)
                break
        else:
            inner = getattr(item, "element", None)
            rows.append(inner if isinstance(inner, dict) else {"label": str(item)})
    return json.dumps(rows, default=str, separators=(",", ":"))


def make_llm(model: str | None = None, tools: list[dict] | None = None) -> Callable:
    """Build the decision callable used by `agent.run`.

    Raises `RuntimeError` when no API key is configured: callers that want an
    offline run should use `agentic.offline.make_offline_llm` instead.
    """
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError

    m = model or SETTINGS.model
    tool_schemas = tools if tools is not None else TOOLS
    client = OpenAI(api_key=SETTINGS.openai_api_key, timeout=SETTINGS.timeout)

    def decide(goal: str, url: str, elements, history: list, context: dict | None = None) -> dict:
        hist = "\n".join(f"- {h.action}: {h.result}" for h in list(history)[-5:])
        elem_txt = _serialise_elements(elements)
        prompt = (
            f"Goal: {goal}\nURL: {url}\n"
            f"Retrieved elements (most relevant first):\n{elem_txt}\n"
            f"History:\n{hist}"
        )
        if context:
            prompt += f"\nAgent state: {json.dumps(context, default=str, separators=(',', ':'))}"
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You control a browser. Pick exactly one tool call. "
                                "Always target an element from the retrieved list by its selector. "
                                "If the same state was already visited and the last action did not help, "
                                "change strategy instead of repeating it."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tools=tool_schemas,
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
