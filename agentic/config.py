"""Central configuration from env / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("AGENT_MODEL", "gpt-4o-mini"))
    max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "10")))
    timeout: int = field(default_factory=lambda: int(os.environ.get("AGENT_TIMEOUT", "30")))
    max_actions_per_min: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_ACTIONS_PER_MIN", "30")))
    #: Scrolling is cheap but abusable: a model with nothing else to do can
    #: spend the whole action budget spinning the wheel. It gets its own cap.
    max_scrolls_per_min: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_SCROLLS_PER_MIN", "12")))
    #: Number of elements handed to the model per step after retrieval.
    top_k: int = field(default_factory=lambda: int(os.environ.get("AGENT_TOP_K", "8")))
    #: `1` to watch the browser (`--headed` overrides this).
    headed: bool = field(default_factory=lambda: _truthy(os.environ.get("AGENT_HEADED")))
    #: Where the trained selector-ranking model is persisted. Missing file is a
    #: supported state: the agent falls back to the heuristic ranking.
    selector_model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AGENT_SELECTOR_MODEL") or (_repo_root() / "models" / "selector_ranker.joblib")
        )
    )
    #: Default location of the episodic-memory / feedback SQLite database.
    memory_db: str = field(default_factory=lambda: os.environ.get("AGENT_MEMORY_DB", ""))
    allowed_domains: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            d.strip() for d in os.environ.get("AGENT_ALLOWED_DOMAINS", "example.com").split(",") if d.strip()
        )
    )

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key)


SETTINGS = Settings()
