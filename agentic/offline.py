"""Offline decision policy: the browser-free, key-free brain for demos and CI.

Read this before using it. `OfflinePlanner` is **not** a language model and does
not pretend to be: it is a deterministic argmax over what the perception layer
already computed. It exists for three honest reasons:

1. The live CLI (`python -m agentic`) previously crashed without an API key
   because `__main__` passed `llm=None` into the loop.
2. The agent loop needs a planner that runs in CI, offline, reproducibly.
3. It is the baseline the LLM has to beat. If a lexical ranker plus an argmax
   gets the goal done, the model is not earning its latency and cost.

The policy:
- take the highest-scoring element the retrieval already ranked,
- skip anything the trace shows was tried at this state,
- type when the goal contains a payload (a quoted string, an email, a coupon
  code, "search for X") and the shortlist has a field to put it in,
- otherwise click, and declare `done` when the shortlist has nothing new.

Everything it decides is recorded in `self.log` with the reason, so a trace can
be read by a reviewer without guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .logging_config import get_logger
from .snapshot import normalize_text

log = get_logger(__name__)

#: Roles that accept text. `type` is only chosen when one of these is offered.
FIELD_ROLES = frozenset({"textbox", "searchbox", "combobox", "input", "listbox", "slider"})

_QUOTED_RE = re.compile(r"[\"']([^\"']{1,80})[\"']")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
_SEARCH_RE = re.compile(r"(?:search(?:\s+for)?|find|look\s+up)\s+(.{2,80})", re.I)
_ENTER_RE = re.compile(r"(?:type|enter|fill(?:\s+in)?|write)\s+(.{1,80})", re.I)
_CODE_RE = re.compile(r"(?:coupon|promo(?:\s+code)?|voucher|code)\s+([A-Za-z0-9-]{3,20})")

#: Verbs that mean "navigate" when the shortlist has nothing to act on.
_URL_RE = re.compile(r"https?://[^\s\"']+", re.I)


def goal_url(goal: str) -> str:
    """The literal URL inside a goal, or "" when it has none."""
    match = _URL_RE.search(normalize_text(goal))
    return match.group(0).rstrip(".,") if match else ""


def goal_payload(goal: str) -> str:
    """Text the goal wants *entered*, or "" when the goal has none.

    Ordering is deliberate: an explicit quote beats a regex guess, because
    `"Search for the best coupon code"` means search for that literal string.
    """
    text = normalize_text(goal)
    for matcher in (_QUOTED_RE, _EMAIL_RE, _CODE_RE, _SEARCH_RE, _ENTER_RE):
        match = matcher.search(text)
        if match:
            value = next((g for g in match.groups() if g), "")
            return value.strip(" .,")
    return ""


@dataclass
class PlannerDecision:
    """One offline decision plus why it was made."""

    action: dict[str, Any]
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trace formatting helper
        return f"{self.action.get('name')} {self.action.get('args')} :: {self.reason}"


@dataclass
class OfflinePlanner:
    """Deterministic stand-in for `make_llm()`.

    Signature-compatible with the real planner: `(goal, url, elements, history,
    context=None) -> {"name": ..., "args": ...}`, so it drops into `agent.run`
    unchanged. It ignores nothing except the actual model call.
    """

    max_actions: int = 12
    #: When the goal carries a payload and the shortlist has an input role, type
    #: it there even if a clickable candidate scores higher. Set False to make
    #: the planner strictly argmax over retrieval.
    prefer_field_for_payload: bool = True
    log: list[PlannerDecision] = field(default_factory=list)
    done_after_success: int = 1

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _field(elements: Sequence[dict]) -> dict | None:
        for element in elements:
            if str(element.get("role", "")).lower() in FIELD_ROLES:
                return dict(element)
        return None

    @staticmethod
    def _ranked(elements: Sequence[dict]) -> list[dict]:
        """Sort by the retrieval score, falling back to the offered order."""
        rows = [dict(e) for e in elements or [] if isinstance(e, dict) and e.get("selector")]
        return sorted(rows, key=lambda e: (-float(e.get("score", 0.0) or 0.0), str(e.get("selector"))))

    @staticmethod
    def _spent(history: Sequence[Any]) -> set[str]:
        """Targets the loop already acted on.

        Read from the trace rather than from private state so the planner stays
        correct across a reused driver and is testable in isolation.
        """
        spent: set[str] = set()
        for step in history or []:
            target = str(getattr(step, "target", "") or "")
            action = str(getattr(step, "action", "") or "").lower()
            if target and action in {"click", "type", "scroll", "goto"}:
                spent.add(target)
        return spent

    # -- planner protocol --------------------------------------------------
    def __call__(
        self,
        goal: str,
        url: str,
        elements: Sequence[Any],
        history: Sequence[Any] = (),
        context: dict | None = None,
    ) -> dict:
        decision = self.decide(goal, url, elements, history, context)
        return decision.action

    def decide(
        self,
        goal: str,
        url: str,
        elements: Sequence[Any],
        history: Sequence[Any] = (),
        context: dict | None = None,
    ) -> PlannerDecision:
        rows = self._ranked(self._as_dicts(elements))
        steps = list(history or ())
        spent = self._spent(steps)
        fresh = [r for r in rows if str(r["selector"]) not in spent]
        payload = goal_payload(goal)
        successes = sum(
            1 for s in steps if str(getattr(s, "reason", "")).startswith(("executed", "loop-break"))
        )

        # 1. our own budget: the loop has one, but a reused planner must not spin
        #    across sessions either.
        if len(steps) >= self.max_actions:
            return self._record(
                {"name": "done", "args": {}},
                f"planner budget of {self.max_actions} step(s) reached",
            )

        # 2. an explicit destination outranks anything on the current page.
        wanted = goal_url(goal)
        if wanted and wanted not in spent:
            return self._record(
                {"name": "goto", "args": {"url": wanted}},
                f"goal names an unvisited url ({wanted})",
            )

        # 3. nothing left to try: stop and say why, rather than re-offering.
        if not fresh:
            action = {"name": "done", "args": {}}
            if not rows:
                reason = (
                    "retrieval offered no elements"
                    if payload == ""
                    else f"no input offered for payload {payload!r}"
                )
            else:
                reason = f"nothing new: {len(rows)} offered, {len(spent)} already spent"
            return self._record(action, reason)

        top = fresh[0]
        selector = str(top["selector"])
        role = str(top.get("role", "")).lower()
        score = float(top.get("score", 0.0) or 0.0)

        # 4. the goal carries text to enter and a field exists: type beats click.
        field_element = self._field(fresh)
        if payload and (role in FIELD_ROLES or (self.prefer_field_for_payload and field_element)):
            target = top if role in FIELD_ROLES else field_element
            return self._record(
                {"name": "type", "args": {"selector": str(target["selector"]), "text": payload}},
                f"payload {payload!r} -> type into {str(target.get('role','') or 'field')} "
                f"{target['selector']!r} (top fresh was {selector!r}, score={score:.3f})",
            )

        # 5. objective verb already landed with effect and nothing more is asked.
        if successes >= self.done_after_success and not payload:
            return self._record(
                {"name": "done", "args": {}},
                f"{successes} action(s) already executed with effect and the goal asks for no more",
            )

        # 6. default: act on the best fresh candidate retrieval produced.
        return self._record(
            {"name": "click", "args": {"selector": selector}},
            f"top fresh candidate {selector!r} (role={role or '?'}, score={score:.3f})",
        )

    # -- plumbing ----------------------------------------------------------
    @staticmethod
    def _as_dicts(elements: Sequence[Any]) -> list[dict]:
        rows: list[dict] = []
        for element in elements or []:
            if isinstance(element, dict):
                rows.append(element)
            elif hasattr(element, "as_dict"):
                rows.append(element.as_dict())  # SelectionHit
            else:
                rows.append({"selector": str(element), "role": "unknown"})
        return rows

    def _record(self, action: dict, reason: str) -> PlannerDecision:
        decision = PlannerDecision(action=action, reason=reason)
        self.log.append(decision)
        if len(self.log) > 200:  # keep a long session bounded
            del self.log[: len(self.log) - 200]
        return decision

    def summary(self) -> list[str]:
        return [f"{i + 1}. {d}" for i, d in enumerate(self.log)]


def make_offline_llm(**kwargs: Any) -> OfflinePlanner:
    """Factory mirroring `llm.make_llm()`, but with no key and no network."""
    return OfflinePlanner(**kwargs)


__all__ = [
    "FIELD_ROLES",
    "OfflinePlanner",
    "PlannerDecision",
    "goal_payload",
    "goal_url",
    "make_offline_llm",
]
