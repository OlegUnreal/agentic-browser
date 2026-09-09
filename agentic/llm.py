"""Real LLM brain for the agentic browser.

Replaces the stub decision parser with structured tool calls via OpenAI.
"""
from __future__ import annotations

import json
import os
from typing import Callable


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


def make_llm(model: str = "gpt-4o-mini") -> Callable:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def decide(goal: str, url: str, elements: str, history: list) -> dict:
        hist = "\n".join(f"- {h.action}: {h.result}" for h in history[-5:])
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You control a browser. Pick exactly one tool call."},
                {"role": "user", "content": f"Goal: {goal}\nURL: {url}\nElements: {elements}\nHistory:\n{hist}"},
            ],
            tools=TOOLS,
            tool_choice="required",
        )
        tc = resp.choices[0].message.tool_calls[0]
        args = json.loads(tc.function.arguments or "{}")
        return {"name": tc.function.name, "args": args}

    return decide
