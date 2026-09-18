"""Learned selector ranking.

A scikit-learn logistic regression over the interpretable features in
`selectors.py`, scored per candidate selector. The point of keeping the feature
vector small and human-readable is that `coefficients()` doubles as a diagnostic
of what the agent has learned to trust — if `css_specificity` comes back
positive, the training data is teaching it something wrong.

Why a linear model rather than a gradient-boosted one: at a few hundred rows per
feature the GBM overfits the synthetic signal and the coefficients stop being
legible. `fit(model="gbm")` is there so the comparison can be made in a test
rather than asserted in prose.

The model is *optional at runtime*: `SelectorRanker.load` returns `None` when no
model file exists, and `rank()` falls back to the documented heuristic. Nothing
in the agent path assumes a trained artifact is present.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..logging_config import get_logger
from .feedback import FeedbackRecord, FeedbackStore
from .selectors import FEATURES, SelectionContext, SelectorCandidate, feature_names, heuristic_scores, rank_candidates

log = get_logger(__name__)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import RidgeClassifier

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

MODEL_KINDS = ("logreg", "ridge", "gbm")


@dataclass
class RankerMetrics:
    """Learned ranking vs the naive 'pick the first candidate' baseline."""

    top1_accuracy: float = 0.0
    mrr: float = 0.0
    baseline_top1_accuracy: float = 0.0
    baseline_mrr: float = 0.0
    n_groups: int = 0
    n_rows: int = 0
    model_kind: str = "heuristic"
    trained_on: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def top1_lift(self) -> float:
        return self.top1_accuracy - self.baseline_top1_accuracy

    @property
    def mrr_lift(self) -> float:
        return self.mrr - self.baseline_mrr

    def as_dict(self) -> dict[str, Any]:
        return {
            "top1_accuracy": round(self.top1_accuracy, 4),
            "mrr": round(self.mrr, 4),
            "baseline_top1_accuracy": round(self.baseline_top1_accuracy, 4),
            "baseline_mrr": round(self.baseline_mrr, 4),
            "top1_lift": round(self.top1_lift, 4),
            "mrr_lift": round(self.mrr_lift, 4),
            "n_groups": self.n_groups,
            "n_rows": self.n_rows,
            "model_kind": self.model_kind,
            "trained_on": self.trained_on,
        }

    def __str__(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


def _split_groups(groups: Sequence[str], test_fraction: float = 0.3, salt: str = "selector") -> tuple[set[str], set[str]]:
    """Deterministic group-wise train/test split.

    Splitting by group (one page-state decision) rather than by row is
    mandatory: every row of a group shares the same shortlist, so a row-wise
    split would put the same decision in both folds and inflate the score.

    The bucket comes from blake2b, not `hash()`: the builtin is salted per
    process, which would make the fold assignment (and therefore the reported
    metrics) change between runs.
    """
    test: set[str] = set()
    train: set[str] = set()
    for group in sorted(set(groups)):
        digest = hashlib.blake2b(f"{salt}|{group}".encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % 1000
        if bucket / 1000.0 < test_fraction:
            test.add(group)
        else:
            train.add(group)
    if not test or not train:  # degenerate corpus: report in-sample rather than crash
        return set(groups), set(groups)
    return train, test


def split_rows(
    records: Sequence[FeedbackRecord] | FeedbackStore,
    test_fraction: float = 0.3,
    salt: str = "selector",
) -> tuple[list[FeedbackRecord], list[FeedbackRecord]]:
    """Group-wise (train, test) split of labelled rows.

    Exposed publicly so a honest evaluation is the easy path: fit on the train
    fold, score the test fold, and no shortlist ever appears in both.
    """
    rows = records.load() if isinstance(records, FeedbackStore) else list(records)
    train_groups, test_groups = _split_groups([r.group_id for r in rows], test_fraction, salt=salt)
    train = [r for r in rows if r.group_id in train_groups]
    test = [r for r in rows if r.group_id in test_groups]
    return train, test


def _first_relevant_rank(labels: Sequence[int]) -> int:
    for position, label in enumerate(labels, start=1):
        if label == 1:
            return position
    return 0


def ranking_metrics(
    groups: Sequence[Sequence[tuple[int, float]]],
) -> tuple[float, float, float, float]:
    """(top1, mrr, baseline_top1, baseline_mrr) over groups of (label, score).

    Baseline is the first row of each group, i.e. the order the candidates were
    offered in — document order in the fixture, which is what an unranked agent
    ships.
    """
    hits = misses = 0
    reciprocal: list[float] = []
    base_hits = 0
    base_reciprocal: list[float] = []
    for rows in groups:
        if not rows:
            continue
        labelled = [label for label, _ in rows if label >= 0]
        if 1 not in labelled:
            continue  # no correct answer known for this group: nothing to score
        misses += 1
        ordered = sorted(rows, key=lambda r: (-r[1], r[0]))
        rank = _first_relevant_rank([label for label, _ in ordered])
        hits += 1 if rank == 1 else 0
        reciprocal.append(1.0 / rank if rank else 0.0)
        base_rank = _first_relevant_rank([label for label, _ in rows])
        base_hits += 1 if base_rank == 1 else 0
        base_reciprocal.append(1.0 / base_rank if base_rank else 0.0)
    total = max(misses, 1)
    return hits / total, sum(reciprocal) / total, base_hits / total, sum(base_reciprocal) / total


class SelectorRanker:
    """Wraps a fitted estimator over `FEATURES`, with a heuristic fallback."""

    def __init__(self, model: Any = None, kind: str = "logreg", trained_on: str = ""):
        self.model = model
        self.kind = kind if model is not None else "heuristic"
        self.trained_on = trained_on
        self.feature_names = feature_names()

    # -- construction ------------------------------------------------------
    @property
    def is_learned(self) -> bool:
        return self.model is not None

    @staticmethod
    def _make_estimator(kind: str):
        if not SKLEARN_AVAILABLE:
            return None
        if kind == "ridge":
            return RidgeClassifier(alpha=1.0, class_weight="balanced")
        if kind == "gbm":
            return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, random_state=0)
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")

    def fit(self, records: Sequence[FeedbackRecord] | FeedbackStore, kind: str = "logreg") -> "SelectorRanker":
        rows = records.load() if isinstance(records, FeedbackStore) else list(records)
        rows = [r for r in rows if len(r.features) == len(FEATURES) and r.label in (0, 1)]
        if len(rows) < 20 or len({r.label for r in rows}) < 2:
            raise ValueError(
                f"need >=20 labelled rows spanning both outcomes, got {len(rows)} rows "
                f"({sorted({r.label for r in rows})})"
            )
        estimator = self._make_estimator(kind)
        if estimator is None:
            raise RuntimeError("scikit-learn is required to fit the selector ranker")
        X = np.asarray([r.features for r in rows], dtype=np.float64)
        y = np.asarray([r.label for r in rows], dtype=int)
        estimator.fit(X, y)
        self.model = estimator
        self.kind = kind
        self.trained_on = f"{len(rows)} rows / {len({r.group_id for r in rows})} groups"
        log.info("ranker_fitted", extra={"kind": kind, "rows": len(rows)})
        return self

    @classmethod
    def fit_from(cls, records, kind: str = "logreg") -> "SelectorRanker":
        return cls().fit(records, kind=kind)

    # -- inference ---------------------------------------------------------
    def scores(self, feature_rows: Sequence[Sequence[float]]) -> list[float]:
        if not len(list(feature_rows)):
            return []
        if self.model is None:
            raise RuntimeError("scores() requires a fitted model; use rank() for the fallback")
        X = np.asarray(list(feature_rows), dtype=np.float64)
        if hasattr(self.model, "predict_proba"):
            try:
                return [float(p[1]) for p in self.model.predict_proba(X)]
            except Exception:  # noqa: BLE001 - odd sklearn output shapes
                pass
        decision = getattr(self.model, "decision_function", None)
        if callable(decision):
            values = np.asarray(decision(X), dtype=float).ravel()
            return [float(1.0 / (1.0 + math.exp(-v))) for v in values]
        return [float(v) for v in np.asarray(self.model.predict(X), dtype=float).ravel()]

    def predict(self, feature_rows: Sequence[Sequence[float]]) -> list[float]:
        """Public scoring hook: probability-like score in ascending-good order."""
        if self.model is None:
            return [float(v) for v in _heuristic_from_features(feature_rows)]
        return self.scores(feature_rows)

    def rank(
        self,
        candidates: Sequence[SelectorCandidate],
        context: SelectionContext | None = None,
    ) -> list[SelectorCandidate]:
        """Rank a shortlist. Falls back to the heuristic when untrained."""
        context = context or SelectionContext()
        if not self.is_learned:
            return rank_candidates(candidates, context, predict=None)
        return rank_candidates(candidates, context, predict=self.predict)

    def coefficients(self) -> dict[str, float]:
        """Per-feature weights — the legible part of the learned policy."""
        if self.model is None:
            return {}
        try:
            coef = np.asarray(self.model.coef_, dtype=float).ravel()
        except AttributeError:
            return {}
        return {name: round(float(value), 4) for name, value in zip(FEATURES, coef[: len(FEATURES)])}

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "kind": self.kind, "features": list(FEATURES)}, target)
        manifest = target.with_suffix(target.suffix + ".meta.json")
        manifest.write_text(
            json.dumps({"kind": self.kind, "features": list(FEATURES), "trained_on": self.trained_on}, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path | None, expected_features: Sequence[str] = FEATURES) -> "SelectorRanker | None":
        """`None` when absent or incompatible — the documented fallback signal.

        A feature-set mismatch is treated as absent rather than loaded, because
        scoring stale features silently would rank on scrambled inputs.
        """
        if not path:
            return None
        target = Path(path)
        if not target.exists():
            log.info("ranker_missing", extra={"path": str(target), "action": "using heuristic ranking"})
            return None
        try:
            import joblib

            payload = joblib.load(target)
        except Exception as exc:  # noqa: BLE001
            log.warning("ranker_load_failed", extra={"error": type(exc).__name__})
            return None
        if isinstance(payload, dict):
            features = tuple(payload.get("features", ()))
            if features != tuple(expected_features):
                log.warning("ranker_feature_drift", extra={"stored": len(features), "expected": len(expected_features)})
                return None
            model = payload.get("model")
            kind = str(payload.get("kind", "logreg"))
        else:
            model, kind = payload, "logreg"
        if model is None:
            return None
        return cls(model=model, kind=kind, trained_on=f"loaded:{target.name}")

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> "SelectorRanker":
        return cls.load(path) or cls()

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        records: Sequence[FeedbackRecord] | FeedbackStore,
        split: bool = True,
        test_fraction: float = 0.3,
    ) -> RankerMetrics:
        rows = records.load() if isinstance(records, FeedbackStore) else list(records)
        rows = [r for r in rows if len(r.features) == len(FEATURES)]
        if not rows:
            return RankerMetrics(model_kind=self.kind)
        train_groups, test_groups = _split_groups([r.group_id for r in rows], test_fraction)
        evaluation = [r for r in rows if (r.group_id in test_groups if split else True)]
        if not evaluation:
            evaluation = rows
        by_group: dict[str, list[FeedbackRecord]] = {}
        for row in evaluation:
            by_group.setdefault(row.group_id, []).append(row)
        scores = self.scores([r.features for r in evaluation]) if self.is_learned else _heuristic_from_features(
            [r.features for r in evaluation]
        )
        score_of = {id(row): float(value) for row, value in zip(evaluation, scores)}
        top1, mrr, base_top1, base_mrr = ranking_metrics(
            [[(row.label, score_of[id(row)]) for row in group_rows] for group_rows in by_group.values()]
        )
        if self.is_learned:
            heuristic_of = {
                id(row): float(value)
                for row, value in zip(evaluation, _heuristic_from_features([r.features for r in evaluation]))
            }
            heur_top1, heur_mrr, _, _ = ranking_metrics(
                [[(row.label, heuristic_of[id(row)]) for row in group_rows] for group_rows in by_group.values()]
            )
        else:
            heur_top1, heur_mrr = top1, mrr
        return RankerMetrics(
            top1_accuracy=top1,
            mrr=mrr,
            baseline_top1_accuracy=base_top1,
            baseline_mrr=base_mrr,
            n_groups=len(by_group),
            n_rows=len(evaluation),
            model_kind=self.kind if self.is_learned else "heuristic",
            trained_on=self.trained_on or "untrained",
            detail={
                "train_groups": len(train_groups & {r.group_id for r in rows}),
                "test_groups": len(test_groups),
                "heuristic_top1": round(heur_top1, 4),
                "heuristic_mrr": round(heur_mrr, 4),
            },
        )


def _heuristic_from_features(feature_rows: Sequence[Sequence[float]]) -> list[float]:
    """Same formula as `selectors.heuristic_scores`, from a feature vector.

    Needed at evaluation time because the fixture stores features, not the
    candidate objects the heuristic scorer walks.
    """
    index = {name: position for position, name in enumerate(FEATURES)}
    out: list[float] = []
    for row in feature_rows:
        if len(row) != len(FEATURES):
            out.append(0.0)
            continue
        get = lambda name: float(row[index[name]])  # noqa: E731 - local view helper
        out.append(
            round(
                2.0 * get("text_match")
                + 0.45 * get("prior_success_rate")
                + 0.25 * get("role_actionable")
                + 0.20 * get("visible")
                + 0.10 * get("in_viewport")
                - 0.30 * get("css_specificity")
                - 0.10 * get("norm_doc_order"),
                6,
            )
        )
    return out


__all__ = ["RankerMetrics", "SelectorRanker", "ranking_metrics", "split_rows", "MODEL_KINDS", "SKLEARN_AVAILABLE"]
