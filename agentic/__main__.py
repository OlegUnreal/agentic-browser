"""CLI: python -m agentic "<goal>" "<start_url>"""
from __future__ import annotations

import argparse
import sys

from .config import SETTINGS
from .logging_config import setup_logging, get_logger
from .agent import run

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="agentic", description="Agentic browser")
    parser.add_argument("goal", help="What the agent should accomplish")
    parser.add_argument("start_url", help="Starting URL")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    if not SETTINGS.has_key:
        log.error("missing_api_key", extra={"hint": "set OPENAI_API_KEY or create a .env"})
        return 2

    try:
        steps = run(
            args.goal,
            args.start_url,
            None,  # type: ignore[arg-type]
            max_steps=args.max_steps or SETTINGS.max_steps,
            allowed_domains=set(SETTINGS.allowed_domains),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("agent_failed", extra={"error": type(exc).__name__})
        return 1

    for s in steps:
        print(f"{s.n}. {s.action} -> {s.result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
