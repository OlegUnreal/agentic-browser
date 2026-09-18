"""Episodic page memory.

Two jobs, both about the same failure mode — an agent that keeps doing the
thing that already did not work:

1. **Loop breaking.** Every visited state gets a structural signature (URL +
   ordered interactive-role shape, see `snapshot.structure_hash`). When the
   signature comes back a second time, the memory says so, names the action
   that caused the revisit, and the loop forces a different one.
2. **Recall.** What was tried here before and did it work? Fed into the
   planner as text, and used as the prior-success feature by the selector
   ranker.

Backed by SQLite so it survives a process restart when pointed at a file, and
runs entirely in RAM (`:memory:`) in tests. All reads/writes are ordered by an
explicit rowid and every query is deterministic; no wall-clock value is ever
part of a decision path, only stored for observability.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..snapshot import normalize_url, selector_shape, structure_hash

SCHEMA_VERSION = 1

_OUTCOME_OK = "ok"
_OUTCOME_BAD = ("failed", "blocked", "rate-limited", "no-effect")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS states (
    state_hash   TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    n_elements   INTEGER NOT NULL,
    first_seen   REAL NOT NULL,
    last_seen    REAL NOT NULL,
    visits       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash   TEXT NOT NULL,
    goal_hash    TEXT NOT NULL,
    step         INTEGER NOT NULL,
    action       TEXT NOT NULL,
    target       TEXT NOT NULL,
    args_json    TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    detail       TEXT NOT NULL,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS episodes_state ON episodes(state_hash, id);
CREATE INDEX IF NOT EXISTS episodes_goal  ON episodes(goal_hash, id);
CREATE TABLE IF NOT EXISTS selections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash    TEXT NOT NULL,
    goal_hash     TEXT NOT NULL,
    elements_json TEXT NOT NULL,
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS selections_state ON selections(state_hash, id);
"""


@dataclass
class Episode:
    """One recorded action at one state."""

    state_hash: str
    goal_hash: str
    step: int
    action: str
    target: str
    outcome: str
    detail: str = ""
    args: dict = field(default_factory=dict)
    rowid: int = -1

    @property
    def ok(self) -> bool:
        return self.outcome == _OUTCOME_OK

    def key(self) -> str:
        return action_key(self.action, self.target, self.args)

    def describe(self) -> str:
        target = f" {self.target}" if self.target else ""
        return f"step {self.step}: {self.action}{target} -> {self.outcome} ({self.detail})"


@dataclass
class StateSignature:
    """What the memory knows about a state at the moment it is observed."""

    state_hash: str
    url: str
    visits: int
    elements: int
    prior_episodes: list[Episode] = field(default_factory=list)
    #: This signature has been seen before, so the last action did not change
    #: the page in any structurally meaningful way.
    repeat: bool = False
    #: The action that, last time we were here, produced this same state.
    caused_by: str = ""

    @property
    def tried(self) -> set[str]:
        return {e.key() for e in self.prior_episodes}

    @property
    def failed(self) -> set[str]:
        return {e.key() for e in self.prior_episodes if not e.ok}

    def worked(self) -> list[Episode]:
        return [e for e in self.prior_episodes if e.ok]

    def hint(self) -> str:
        """Planner-facing text: what worked and what is already spent here."""
        bits: list[str] = []
        if self.visits > 1:
            bits.append(f"state revisited {self.visits}x (loop risk)")
        if self.caused_by:
            bits.append(f"last arrival caused by: {self.caused_by}")
        for episode in self.prior_episodes[-4:]:
            bits.append(episode.describe())
        return "; ".join(bits)


@dataclass
class LoopWarning:
    """Emitted when a proposed action would repeat a known-dead end."""

    kind: str
    reason: str
    action_key_name: str
    state_hash: str
    visits: int
    alternatives: list[str] = field(default_factory=list)


def action_key(action: str, target: str = "", args: dict | None = None) -> str:
    """Canonical identity of a decision: verb + generalised target (+ payload).

    `type` keys include the text, because typing "a@b.com" then "c@d.com" is
    genuinely new information, whereas re-clicking the same node is not.
    """
    name = str(action or "").lower()
    args = args or {}
    if name == "type":
        return f"type|{selector_shape(target)}|{str(args.get('text', ''))[:64]}"
    if name == "goto":
        return f"goto|{normalize_url(str(args.get('url', '')))[:160]}"
    if name == "scroll":
        return f"scroll|{selector_shape(target)}|{int(args.get('amount', 500) or 0)}"
    if target:
        return f"{name}|{selector_shape(target)}"
    return name


def goal_hash(goal: str) -> str:
    return structure_hash("", [], goal)[:16]


class EpisodicMemory:
    """SQLite-backed store of visited states, actions and outcomes."""

    def __init__(self, path: str | None = ":memory:", max_episodes: int = 5000, clock=time.time):
        self.path = path or ":memory:"
        self.max_episodes = int(max_episodes)
        self._clock = clock
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
        self._db.commit()

    # -- plumbing ----------------------------------------------------------
    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "EpisodicMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def commit(self) -> None:
        self._db.commit()

    def _trim(self) -> None:
        """Bound the store: drop the oldest episodes past `max_episodes`."""
        cur = self._db.execute("SELECT COUNT(*) AS n FROM episodes")
        count = int(cur.fetchone()["n"])
        if count > self.max_episodes:
            self._db.execute(
                "DELETE FROM episodes WHERE id IN "
                "(SELECT id FROM episodes ORDER BY id ASC LIMIT ?)",
                (count - self.max_episodes,),
            )

    # -- write path --------------------------------------------------------
    def observe(
        self,
        state_hash: str,
        url: str,
        elements: Sequence[dict] | None = (),
        goal: str = "",
        title: str = "",
    ) -> StateSignature:
        """Record that we are looking at this state and return what we know."""
        now = float(self._clock())
        ghash = goal_hash(goal)
        n_elements = len(elements or ())
        row = self._db.execute("SELECT visits FROM states WHERE state_hash = ?", (state_hash,)).fetchone()
        visits = int(row["visits"]) + 1 if row else 1
        if row:
            self._db.execute(
                "UPDATE states SET last_seen = ?, visits = ? WHERE state_hash = ?",
                (now, visits, state_hash),
            )
        else:
            self._db.execute(
                "INSERT INTO states(state_hash, url, title, n_elements, first_seen, last_seen, visits)"
                " VALUES(?,?,?,?,?,?,1)",
                (state_hash, normalize_url(url), str(title or "")[:120], n_elements, now, now),
            )
        episodes = self.recall(state_hash=state_hash, goal_hash_value=ghash)
        caused_by = episodes[-1].key() if episodes and visits > 1 else ""
        self._db.commit()
        return StateSignature(
            state_hash=state_hash,
            url=normalize_url(url),
            visits=visits,
            elements=n_elements,
            prior_episodes=episodes,
            repeat=visits > 1,
            caused_by=caused_by,
        )

    def record_action(
        self,
        state_hash: str,
        goal: str,
        step: int,
        action: str,
        target: str = "",
        args: dict | None = None,
        outcome: str = _OUTCOME_OK,
        detail: str = "",
    ) -> Episode:
        ghash = goal_hash(goal)
        args = args or {}
        now = float(self._clock())
        cur = self._db.execute(
            "INSERT INTO episodes(state_hash, goal_hash, step, action, target, args_json, outcome, detail, ts)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                state_hash,
                ghash,
                int(step),
                str(action or "").lower(),
                str(target or ""),
                json.dumps(args, sort_keys=True, default=str)[:2000],
                str(outcome),
                str(detail or "")[:400],
                now,
            ),
        )
        self._trim()
        self._db.commit()
        return Episode(
            state_hash=state_hash,
            goal_hash=ghash,
            step=int(step),
            action=str(action or "").lower(),
            target=str(target or ""),
            outcome=str(outcome),
            detail=str(detail or ""),
            args=args,
            rowid=int(cur.lastrowid or -1),
        )

    def note_selection(self, state_hash: str, goal: str, hits: Iterable[dict]) -> None:
        """Persist what retrieval surfaced here, for later offline analysis."""
        payload = [
            {"selector": h.get("selector"), "label": h.get("label"), "role": h.get("role"), "score": h.get("score")}
            for h in list(hits)[:32]
        ]
        self._db.execute(
            "INSERT INTO selections(state_hash, goal_hash, elements_json, ts) VALUES(?,?,?,?)",
            (state_hash, goal_hash(goal), json.dumps(payload, sort_keys=True, default=str)[:8000], float(self._clock())),
        )
        self._db.commit()

    # -- read path ---------------------------------------------------------
    def recall(self, state_hash: str | None = None, goal_hash_value: str | None = None, limit: int = 12) -> list[Episode]:
        """Episodes at a state, oldest first — deterministic ordering by rowid."""
        clauses, params = [], []
        if state_hash:
            clauses.append("state_hash = ?")
            params.append(state_hash)
        if goal_hash_value:
            clauses.append("goal_hash = ?")
            params.append(goal_hash_value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.execute(
            f"SELECT * FROM episodes {where} ORDER BY id ASC LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
        return [
            Episode(
                state_hash=r["state_hash"],
                goal_hash=r["goal_hash"],
                step=int(r["step"]),
                action=r["action"],
                target=r["target"],
                outcome=r["outcome"],
                detail=r["detail"],
                args=_load_json(r["args_json"]),
                rowid=int(r["id"]),
            )
            for r in rows
        ]

    def recall_for_goal(self, goal: str, limit: int = 12) -> list[Episode]:
        return self.recall(goal_hash_value=goal_hash(goal), limit=limit)

    def visits(self, state_hash: str) -> int:
        row = self._db.execute("SELECT visits FROM states WHERE state_hash = ?", (state_hash,)).fetchone()
        return int(row["visits"]) if row else 0

    def outcomes_by_target(self, goal: str | None = None) -> dict[str, dict[str, int]]:
        """`{action_key: {"ok": n, "bad": n}}` — the prior-success source."""
        if goal:
            rows = self._db.execute(
                "SELECT action, target, args_json, outcome FROM episodes WHERE goal_hash = ? ORDER BY id",
                (goal_hash(goal),),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT action, target, args_json, outcome FROM episodes ORDER BY id"
            ).fetchall()
        table: dict[str, dict[str, int]] = {}
        for row in rows:
            key = action_key(row["action"], row["target"], _load_json(row["args_json"]))
            bucket = table.setdefault(key, {"ok": 0, "bad": 0})
            bucket["ok" if row["outcome"] == _OUTCOME_OK else "bad"] += 1
        return table

    def prior_success(self, action: str, target: str = "", args: dict | None = None, goal: str | None = None) -> tuple[int, int]:
        """(successes, attempts) for an action key, smoothed by the caller."""
        table = self.outcomes_by_target(goal)
        bucket = table.get(action_key(action, target, args), {"ok": 0, "bad": 0})
        return bucket["ok"], bucket["ok"] + bucket["bad"]

    def detect_loop(self, signature: StateSignature, action: str, target: str = "", args: dict | None = None) -> LoopWarning | None:
        """Should this action be refused because we are going in circles?

        Fires on the instructed condition — same state signature twice, same
        decision — and on the milder case where the state is new to this step
        but the exact action already failed here.
        """
        key = action_key(action, target, args)
        if not signature.repeat:
            if key in signature.failed:
                return LoopWarning(
                    kind="repeated-failure",
                    reason=f"'{key}' already failed at this state",
                    action_key_name=key,
                    state_hash=signature.state_hash,
                    visits=signature.visits,
                )
            return None
        if key in signature.tried:
            return LoopWarning(
                kind="state-loop",
                reason=(
                    f"state {signature.state_hash[:8]} visited {signature.visits}x and '{key}' was already "
                    f"tried here; repeating it cannot change the page"
                ),
                action_key_name=key,
                state_hash=signature.state_hash,
                visits=signature.visits,
            )
        return None

    def stats(self) -> dict[str, Any]:
        states = int(self._db.execute("SELECT COUNT(*) AS n FROM states").fetchone()["n"])
        episodes = int(self._db.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"])
        selections = int(self._db.execute("SELECT COUNT(*) AS n FROM selections").fetchone()["n"])
        return {"states": states, "episodes": episodes, "selections": selections, "path": self.path}


def _load_json(raw: Any) -> dict:
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


__all__ = [
    "Episode",
    "EpisodicMemory",
    "LoopWarning",
    "StateSignature",
    "action_key",
    "goal_hash",
    "SCHEMA_VERSION",
]
