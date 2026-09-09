"""Central configuration from env / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("AGENT_MODEL", "gpt-4o-mini"))
    max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "10")))
    timeout: int = field(default_factory=lambda: int(os.environ.get("AGENT_TIMEOUT", "30")))
    max_actions_per_min: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_ACTIONS_PER_MIN", "30")))
    allowed_domains: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            d.strip() for d in os.environ.get("AGENT_ALLOWED_DOMAINS", "example.com").split(",") if d.strip()
        )
    )

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key)


SETTINGS = Settings()
