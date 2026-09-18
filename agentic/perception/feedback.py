"""Feedback log: every (candidate selector, context features, outcome).

This is the training corpus the agent grows while running. Written as a flat
append-only SQLite table, one row per *candidate that was offered* — not just
the chosen one — because a ranker trained only on winners has no negatives and
learns nothing about the selectors that were skipped.

`label` is the outcome of acting on that candidate: 1 resolved the step,
0 failed / was refused / had no effect. Rows are grouped by `group_id`
(state signature + goal + step) so train/test splits can be made by group and
the prior-success feature can be computed leave-one-out instead of leaking the
label of the very row being scored.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..snapshot import selector_shape
from .selectors import FEATURES, SelectionContext, SelectorCandidate, featurise

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id     TEXT NOT NULL,
    state_hash   TEXT NOT NULL,
    goal         TEXT NOT NULL,
    action       TEXT NOT NULL,
    selector     TEXT NOT NULL,
    origin       TEXT NOT NULL,
    role         TEXT NOT NULL,
    offered_rank INTEGER NOT NULL,
    chosen       INTEGER NOT NULL,
    label        INTEGER NOT NULL,
    outcome      TEXT NOT NULL,
    features_json TEXT NOT NULL,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_group ON feedback(group_id, id);
CREATE INDEX IF NOT EXISTS feedback_sel   ON feedback(selector, label);
"""


@dataclass
class FeedbackRecord:
    group_id: str
    state_hash: str
    goal: str
    action: str
    selector: str
    origin: str
    role: str
    offered_rank: int
    chosen: bool
    label: int
    outcome: str
    features: list[float]
    rowid: int = -1

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "state_hash": self.state_hash,
            "goal": self.goal,
            "action": self.action,
            "selector": self.selector,
            "origin": self.origin,
            "role": self.role,
            "offered_rank": self.offered_rank,
            "chosen": int(self.chosen),
            "label": int(self.label),
            "outcome": self.outcome,
            "features": list(self.features),
        }


@dataclass
class TrainingSet:
    """Grouped arrays ready for scikit-learn."""

    X: list[list[float]] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    records: list[FeedbackRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def positive_groups(self) -> int:
        return len({g for g, label in zip(self.groups, self.y) if label == 1})


def make_group_id(state_hash: str, goal: str, step: int, action: str) -> str:
    return f"{state_hash[:12]}|{goal[:24]}|{step}|{action}"


class FeedbackStore:
    """SQLite feedback log with typed read/write helpers."""

    def __init__(self, path: str | None = ":memory:", clock=time.time):
        self.path = path or ":memory:"
        self._clock = clock
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "FeedbackStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"])

    # -- write -------------------------------------------------------------
    def record_offer(
        self,
        candidates: Sequence[SelectorCandidate],
        context: SelectionContext,
        state_hash: str,
        step: int,
        chosen_selector: str = "",
    ) -> int:
        """Log the whole shortlist; the chosen one is the only labelled row yet.

        Unchosen rows are written with `label=0` only once the step outcome is
        known (see :meth:`record_outcome`); until then they stay `label=-1` and
        are excluded from training, which keeps "not selected" from being
        conflated with "selected and failed".
        """
        group = make_group_id(state_hash, context.goal, step, context.action)
        rows = 0
        for rank, candidate in enumerate(candidates):
            features = candidate.features or featurise(candidate, context)
            self._db.execute(
                "INSERT INTO feedback(group_id, state_hash, goal, action, selector, origin, role, offered_rank,"
                " chosen, label, outcome, features_json, ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    group,
                    state_hash,
                    context.goal[:200],
                    context.action,
                    candidate.selector[:300],
                    candidate.origin,
                    str(candidate.element.get("role", "unknown")),
                    rank,
                    int(candidate.selector == chosen_selector),
                    -1,
                    "pending",
                    json.dumps([round(float(f), 6) for f in features]),
                    float(self._clock()),
                ),
            )
            rows += 1
        self._db.commit()
        return rows

    def record_outcome(self, state_hash: str, goal: str, step: int, action: str, chosen_selector: str, outcome: str) -> int:
        """Attach the realised outcome to the offered rows of one group.

        The chosen candidate gets the true label. Rejected candidates become
        negatives *only when the chosen one worked* (a better selector existed in
        the same shortlist, which is the ranking signal); if the chosen one
        failed the group was a dead end and their label is genuinely unknown.
        """
        group = make_group_id(state_hash, goal, step, action)
        label = 1 if outcome == "ok" else 0
        touched = self._db.execute(
            "UPDATE feedback SET label = ?, outcome = ?, chosen = 1 WHERE group_id = ? AND selector = ?",
            (label, outcome, group, chosen_selector[:300]),
        ).rowcount
        self._db.execute(
            "UPDATE feedback SET label = ?, outcome = ? WHERE group_id = ? AND selector <> ? AND label = -1",
            (0 if label == 1 else -1, outcome, group, chosen_selector[:300]),
        )
        self._db.commit()
        return int(touched or 0)

    def add(self, record: FeedbackRecord) -> int:
        cur = self._db.execute(
            "INSERT INTO feedback(group_id, state_hash, goal, action, selector, origin, role, offered_rank,"
            " chosen, label, outcome, features_json, ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.group_id,
                record.state_hash,
                record.goal[:200],
                record.action,
                record.selector[:300],
                record.origin,
                record.role,
                record.offered_rank,
                int(record.chosen),
                int(record.label),
                record.outcome,
                json.dumps([round(float(f), 6) for f in record.features]),
                float(self._clock()),
            ),
        )
        self._db.commit()
        return int(cur.lastrowid or -1)

    def extend(self, records: Iterable[FeedbackRecord]) -> int:
        for record in records:
            self.add(record)
        return len(self)

    # -- read --------------------------------------------------------------
    def load(self, include_pending: bool = False) -> list[FeedbackRecord]:
        where = "" if include_pending else "WHERE label >= 0"
        rows = self._db.execute(f"SELECT * FROM feedback {where} ORDER BY id ASC").fetchall()
        out: list[FeedbackRecord] = []
        for row in rows:
            try:
                features = json.loads(row["features_json"])
            except (json.JSONDecodeError, TypeError):
                features = []
            if not isinstance(features, list) or len(features) != len(FEATURES):
                continue  # schema drift: an old feature set must not poison training
            out.append(
                FeedbackRecord(
                    group_id=row["group_id"],
                    state_hash=row["state_hash"],
                    goal=row["goal"],
                    action=row["action"],
                    selector=row["selector"],
                    origin=row["origin"],
                    role=row["role"],
                    offered_rank=int(row["offered_rank"]),
                    chosen=bool(row["chosen"]),
                    label=int(row["label"]),
                    outcome=row["outcome"],
                    features=[float(f) for f in features],
                    rowid=int(row["id"]),
                )
            )
        return out

    def training_set(self) -> TrainingSet:
        records = self.load()
        return TrainingSet(
            X=[r.features for r in records],
            y=[r.label for r in records],
            groups=[r.group_id for r in records],
            records=records,
        )

    def prior_success_map(self, exclude_group: str | None = None) -> dict[str, tuple[int, int]]:
        """`candidate.key() -> (successes, attempts)` for the ranking feature.

        `exclude_group` drops the group currently being scored, which is what
        makes the feature honest at evaluation time instead of leaking the
        label of the row it is attached to.
        """
        if exclude_group:
            rows = self._db.execute(
                "SELECT selector, origin, label FROM feedback WHERE label >= 0 AND group_id <> ? ORDER BY id",
                (exclude_group,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT selector, origin, label FROM feedback WHERE label >= 0 ORDER BY id"
            ).fetchall()
        table: dict[str, tuple[int, int]] = {}
        for row in rows:
            key = f"{row['origin']}|{selector_shape(row['selector'])}"
            successes, attempts = table.get(key, (0, 0))
            table[key] = (successes + int(row["label"] == 1), attempts + 1)
        return table

    def groups(self) -> list[str]:
        rows = self._db.execute(
            "SELECT group_id FROM feedback GROUP BY group_id ORDER BY MIN(id)"
        ).fetchall()
        return [str(r["group_id"]) for r in rows]


__all__ = [
    "FeedbackRecord",
    "FeedbackStore",
    "TrainingSet",
    "make_group_id",
]
