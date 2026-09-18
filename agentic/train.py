"""Fit, evaluate and persist the selector ranker.

    python -m agentic.train                       # synthetic corpus, 240 groups
    python -m agentic.train --db var/feedback.db  # what the agent actually logged
    python -m agentic.train --model gbm --out models/selector_ranker.joblib

Read the numbers honestly. The default corpus is **synthetic** (see
`perception/corpus.py`): the reported lift measures whether the learner
recovers a known causal signal from the features we expose, not how it performs
on the live web. Point `--db` at a feedback store the agent filled during real
runs and the same command reports the transferable number.

Splitting is by group (`ranker.split_rows`), so a shortlist never appears in
both folds. Two reference points are printed next to the learned score:

* **baseline** — document order, i.e. "use the selector the DOM gave us".
* **heuristic** — the hand-written fallback the agent runs with no model file.

A learned model that cannot beat both is not worth its latency, and this command
is where that would show up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import SETTINGS
from .logging_config import get_logger, setup_logging
from .perception.corpus import build_store, generate_records
from .perception.feedback import FeedbackRecord, FeedbackStore
from .perception.ranker import MODEL_KINDS, RankerMetrics, SelectorRanker, SKLEARN_AVAILABLE, split_rows
from .perception.selectors import FEATURES

log = get_logger(__name__)


def load_records(db: str | None, groups: int, seed: int) -> tuple[list[FeedbackRecord], str]:
    """Labelled rows plus a human-readable description of where they came from."""
    if db:
        path = Path(db)
        if not path.exists():
            raise SystemExit(f"no feedback store at {path}; run the agent with --feedback-log first")
        store = FeedbackStore(path=str(path))
        try:
            rows = store.load()
        finally:
            store.close()
        if len(rows) >= 20:
            return rows, f"sqlite:{path.name}"
        log.warning("feedback_store_thin", extra={"rows": len(rows), "action": "using synthetic corpus"})
    return generate_records(range(groups), seed=seed), f"synthetic:{groups} groups (seed={seed})"


def _kept_groups(test: Sequence[FeedbackRecord]) -> list[FeedbackRecord]:
    """Drop test groups whose shortlist has no positive: nothing to rank there."""
    keep = {r.group_id for r in test if r.label == 1}
    return [r for r in test if r.group_id in keep]


def train(
    groups: int = 240,
    kind: str = "logreg",
    out: str | Path | None = None,
    db: str | None = None,
    test_fraction: float = 0.3,
    seed: int = 17,
    save: bool = True,
) -> tuple[SelectorRanker, RankerMetrics, dict[str, Any]]:
    """Fit on the train fold, evaluate on the unseen fold, persist the artifact."""
    if not SKLEARN_AVAILABLE:  # pragma: no cover - sklearn is an install dependency
        raise SystemExit("scikit-learn is required to train the selector ranker")
    rows, source = load_records(db, groups, seed)
    train_rows, test_rows = split_rows(rows, test_fraction=test_fraction, salt=f"{kind}:{source}")
    test_rows = _kept_groups(test_rows)
    if not train_rows or not test_rows:
        raise SystemExit(
            f"split produced {len(train_rows)} train / {len(test_rows)} test rows; "
            "widen the corpus or lower --test-fraction"
        )

    ranker = SelectorRanker().fit(train_rows, kind=kind)
    metrics = ranker.evaluate(test_rows, split=False)
    metrics.trained_on = f"{source} / {len(train_rows)} train rows"
    metrics.model_kind = kind
    metrics.detail.update(
        {
            "test_fraction": test_fraction,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_groups": len({r.group_id for r in train_rows}),
            "test_groups": len({r.group_id for r in test_rows}),
        }
    )
    payload: dict[str, Any] = {
        "model": str(out) if out else str(SETTINGS.selector_model_path),
        "saved": False,
        "source": source,
        "features": len(FEATURES),
        "metrics": {**metrics.as_dict(), **{k: v for k, v in metrics.detail.items() if k.startswith("heuristic_")}},
        "top_weights": [],
        "sign_check": {},
    }
    if save:
        target = Path(out) if out else SETTINGS.selector_model_path
        payload["model"] = str(ranker.save(target))
        payload["saved"] = True

    coefficients = ranker.coefficients()
    payload["top_weights"] = sorted(coefficients.items(), key=lambda kv: -abs(kv[1]))[:8]
    payload["all_weights"] = coefficients
    brittleness = coefficients.get("css_specificity", 0.0) + coefficients.get("selector_depth", 0.0)
    payload["sign_check"] = {
        # `css_specificity` and `selector_depth` are two normalised views of one
        # construct, so lbfgs is free to put the weight on either. Checking them
        # individually would make the verdict flicker with the regularisation
        # path; the block is the claim we actually want to hold.
        "brittleness_block_negative": brittleness < 0,
        "brittleness_block": round(brittleness, 4),
        "css_specificity": round(coefficients.get("css_specificity", 0.0), 4),
        "selector_depth": round(coefficients.get("selector_depth", 0.0), 4),
        "text_match_positive": coefficients.get("text_match", 0.0) > 0,
        "visible_positive": coefficients.get("visible", 0.0) > 0,
    }
    return ranker, metrics, payload


def format_report(payload: dict[str, Any]) -> str:
    """cp1252-safe text report (no emoji, no fancy dashes)."""
    m = payload["metrics"]
    learned = f"{m['top1_accuracy']:.3f}     {m['mrr']:.3f}"
    baseline = f"{m['baseline_top1_accuracy']:.3f}     {m['baseline_mrr']:.3f}"
    heuristic = (
        f"{m['heuristic_top1']:.3f}     {m['heuristic_mrr']:.3f}"
        if "heuristic_top1" in m
        else "n/a"
    )
    lines = [
        f"model      {payload['model']}{'' if payload['saved'] else ' (not written: dry run)'}",
        f"corpus     {payload['source']} ({payload['features']} features)",
        f"eval       {m['n_rows']} rows in {m['n_groups']} unseen groups",
        "",
        "                 top-1      MRR",
        f"  learned      {learned}",
        f"  heuristic    {heuristic}",
        f"  baseline     {baseline}   (document order)",
        f"  lift         {m['top1_lift']:+.3f}     {m['mrr_lift']:+.3f}   (learned - baseline)",
        "",
        "largest weights",
    ]
    for name, value in payload["top_weights"]:
        lines.append(f"  {value:+.3f}  {name}")
    checks = payload["sign_check"]
    if checks:
        lines.append("")
        lines.append(
            "sign check   brittleness (css_specificity {css:+.3f} + selector_depth {depth:+.3f}) "
            "= {block:+.3f} -> negative: {negative}".format(
                css=checks["css_specificity"],
                depth=checks["selector_depth"],
                block=checks["brittleness_block"],
                negative=checks["brittleness_block_negative"],
            )
        )
        lines.append(
            f"             text_match>0: {checks['text_match_positive']}, visible>0: {checks['visible_positive']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="agentic.train", description="Train the selector ranker")
    parser.add_argument("--groups", type=int, default=240, help="synthetic groups to generate")
    parser.add_argument("--model", choices=MODEL_KINDS, default="logreg")
    parser.add_argument("--out", default=None, help="where to persist (default: config path)")
    parser.add_argument("--db", default=None, help="SQLite feedback store from real runs")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", action="store_true", help="print the metrics payload as JSON")
    parser.add_argument("--dry-run", action="store_true", help="evaluate without writing a model file")
    args = parser.parse_args(argv)

    try:
        _, metrics, payload = train(
            groups=args.groups,
            kind=args.model,
            out=args.out,
            db=args.db,
            test_fraction=args.test_fraction,
            seed=args.seed,
            save=not args.dry_run,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(format_report(payload))
    checks = payload["sign_check"]
    beats_baseline = metrics.top1_accuracy >= metrics.baseline_top1_accuracy
    signs_ok = bool(checks.get("brittleness_block_negative")) and bool(checks.get("text_match_positive"))
    # A model that beats document order by memorising an inverted brittleness
    # weight is worse than no model at all, so the CI gate checks both.
    return 0 if (beats_baseline and signs_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
