"""Live browser agent with a real LLM.

Usage: OPENAI_API_KEY=... python -m agentic.demo_llm "book a table" https://example.com
"""
from __future__ import annotations

import sys

from .agent import run
from .llm import make_llm


def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else "find the contact email"
    url = sys.argv[2] if len(sys.argv) > 2 else "https://example.com"
    for step in run(goal, url, make_llm()):
        print(f"step {step.n}: {step.action} -> {step.result}")


if __name__ == "__main__":
    main()
