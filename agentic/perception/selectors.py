"""Candidate selector generation and interpretable featurisation.

The driver accepts one CSS selector per action, but a snapshot element often
carries several ways to address it: its generated path, its `id`, its `name`
attribute, its visible text. Which of those actually *works* is empirical —
`#login` is robust, `body > div:nth-of-type(4) > button:nth-of-type(2)` breaks
the moment a banner is inserted above it.

So instead of hoping the LLM picks well, we enumerate the candidates, score
them with a learned model (`ranker.py`), and hand the model a ranked shortlist.

Every feature here is deliberately interpretable and computed identically at
training and inference time by the single :func:`featurise` function — a
train/serve skew in this file would silently break the ranker.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..snapshot import clean_label, element_text, normalize_text, selector_shape

#: Where a candidate came from. Order matters: it is the tie-break when scores
#: are equal, so `dom` (the driver's own answer) wins over an inferred variant.
ORIGINS = ("dom", "attr", "text", "history")

_ACTIONABLE_ROLES = frozenset({"button", "link", "submit", "textbox", "searchbox", "combobox", "checkbox", "radio"})
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_STOP = frozenset({"the", "a", "an", "to", "of", "on", "in", "for", "and", "or", "is", "it", "this", "that", "with"})


@dataclass
class SelectionContext:
    """Everything featurisation needs that is not about one candidate."""

    goal: str = ""
    observation: str = ""
    action: str = "click"
    n_candidates: int = 1
    n_elements: int = 1
    viewport_height: int = 900
    prior_success: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def goal_tokens(self) -> frozenset[str]:
        return _tokenise(f"{self.goal} {self.observation}")


def _tokenise(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _WORD_RE.findall(normalize_text(text).lower()) if t not in _STOP and len(t) > 1
    )


def text_match(goal_tokens: frozenset[str], candidate_text: str) -> float:
    """Containment of candidate tokens in the query, Jaccard-blended.

    Containment alone lets a one-word label ("Buy") score like a phrase; Jaccard
    alone punishes long labels. The geometric-ish blend of both is what the
    synthetic corpus in `corpus.py` was shaped against.
    """
    cand_tokens = _tokenise(candidate_text)
    if not goal_tokens or not cand_tokens:
        return 0.0
    overlap = goal_tokens & cand_tokens
    if not overlap:
        return 0.0
    containment = len(overlap) / len(cand_tokens)
    jaccard = len(overlap) / len(goal_tokens | cand_tokens)
    return max(containment, jaccard)


def css_specificity(selector: str) -> float:
    """A cheap, monotone proxy for CSS specificity of the selector itself.

    High specificity here means *brittle addressing*: deep `nth-of-type` chains
    that a layout change invalidates. Real CSS specificity is what the browser
    resolves, not what survives a redeploy, which is why the learned model gets
    to decide how much this matters rather than us hardcoding a penalty.
    """
    s = str(selector or "")
    if not s:
        return 0.0
    ids = s.count("#")
    classes = s.count(".") + s.count("[")
    pseudo = len(re.findall(r":(nth-|not|is|where|first|last)", s))
    depth = max(s.count(">") + s.count(" "), 0)
    return ids * 3.0 + classes * 2.0 + pseudo * 1.5 + depth * 1.0


@dataclass
class SelectorCandidate:
    element: dict
    selector: str
    origin: str = "dom"
    text: str = ""
    score: float = 0.0
    features: list[float] = field(default_factory=list)

    @property
    def role(self) -> str:
        return str(self.element.get("role", "unknown"))

    @property
    def label(self) -> str:
        return str(self.element.get("label", ""))

    def key(self) -> str:
        return f"{self.origin}|{selector_shape(self.selector)}"

    def as_dict(self) -> dict:
        return {
            "selector": self.selector,
            "label": self.label,
            "role": self.role,
            "origin": self.origin,
            "score": round(self.score, 4),
        }


def _variants(element: dict) -> list[tuple[str, str]]:
    selector = str(element.get("selector", ""))
    out: list[tuple[str, str]] = [(selector, "dom")] if selector else []
    attrs = element.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    tag = str(element.get("tag") or "").lower() or _guess_tag(str(element.get("role", "")))
    if attrs.get("id"):
        out.append((f"#{attrs['id']}", "attr"))
    if attrs.get("name"):
        out.append((f'{tag or "input"}[name="{attrs["name"]}"]', "attr"))
    if attrs.get("placeholder"):
        out.append((f'input[placeholder="{attrs["placeholder"]}"]', "attr"))
    label = clean_label(str(element.get("label", "")), 60)
    if label and str(element.get("role", "")) in {"button", "link", "menuitem", "tab"}:
        out.append((f'text="{label}"', "text"))
    return [(s, o) for s, o in out if s]


def _guess_tag(role: str) -> str:
    return {
        "button": "button",
        "link": "a",
        "textbox": "textarea",
        "searchbox": "input",
        "combobox": "select",
        "checkbox": "input",
        "radio": "input",
    }.get(str(role or "").lower(), "")


def build_candidates(
    goal: str,
    elements: Sequence[dict] | None,
    observation: str = "",
    action: str = "click",
    history_targets: Sequence[str] | None = None,
    limit: int = 16,
) -> list[SelectorCandidate]:
    """Expand a retrieved element list into addressable candidates.

    `history_targets` are selectors that worked for this goal before (from
    episodic memory); they enter the pool as `origin="history"` candidates even
    if retrieval did not surface them this step — the whole point of remembering.
    """
    ctx_tokens = _tokenise(f"{goal} {observation}")
    candidates: list[SelectorCandidate] = []
    seen: set[str] = set()
    for element in elements or ():
        for selector, origin in _variants(element):
            key = f"{origin}|{selector}"
            if key in seen:
                continue
            seen.add(key)
            text = element_text(element)
            candidates.append(
                SelectorCandidate(
                    element=element,
                    selector=selector,
                    origin=origin,
                    text=text,
                    score=text_match(ctx_tokens, text),
                )
            )
    for selector in history_targets or ():
        key = f"history|{selector}"
        if not selector or key in seen:
            continue
        seen.add(key)
        candidates.append(
            SelectorCandidate(element={"selector": selector, "role": "unknown"}, selector=selector, origin="history", text=selector)
        )
    # Deterministic pre-ranking so the learned model sees a stable input order:
    # text match first, then origin priority, then the selector string itself.
    candidates.sort(key=lambda c: (-c.score, ORIGINS.index(c.origin) if c.origin in ORIGINS else 9, c.selector))
    return candidates[: int(limit)]


FEATURES: tuple[str, ...] = (
    "text_match",
    "log_label_len",
    "css_specificity",
    "selector_depth",
    "role_actionable",
    "role_button",
    "role_link",
    "role_input",
    "visible",
    "in_viewport",
    "norm_doc_order",
    "norm_position_y",
    "origin_dom",
    "origin_attr",
    "origin_text",
    "origin_history",
    "prior_success_rate",
    "log_prior_attempts",
    "full_text_match",
)


def feature_names() -> tuple[str, ...]:
    return FEATURES


def _rect(element: dict) -> dict:
    rect = element.get("rect") or {}
    return rect if isinstance(rect, dict) else {}


def featurise(candidate: SelectorCandidate, context: SelectionContext) -> list[float]:
    """The single source of truth for the ranking feature vector."""
    element = candidate.element or {}
    role = str(element.get("role", "unknown")).lower()
    rect = _rect(element)
    ctx_tokens = context.goal_tokens
    match = text_match(ctx_tokens, candidate.text or element_text(element))
    label_len = len(str(element.get("label", "") or candidate.text))
    depth = max(str(candidate.selector).count(">") + str(candidate.selector).count(" "), 0)
    n_elements = max(context.n_elements, 1)
    y = float(rect.get("y", 0.0) or 0.0)
    viewport = max(float(context.viewport_height or 900), 1.0)
    successes, attempts = context.prior_success.get(candidate.key(), (0, 0))
    prior_rate = (successes + 1.0) / (attempts + 2.0) if attempts or successes else 0.5
    origin = candidate.origin
    is_top = 1.0 if context.n_candidates and match and match >= 0.999 else 0.0
    return [
        round(match, 6),
        round(math.log1p(label_len) / math.log(81.0), 6),
        round(min(css_specificity(candidate.selector) / 20.0, 1.0), 6),
        round(min(depth / 8.0, 1.0), 6),
        1.0 if role in _ACTIONABLE_ROLES else 0.0,
        1.0 if role == "button" else 0.0,
        1.0 if role == "link" else 0.0,
        1.0 if role in {"textbox", "searchbox", "combobox", "checkbox", "radio", "input"} else 0.0,
        1.0 if element.get("visible", True) else 0.0,
        1.0 if element.get("in_viewport", True) else 0.0,
        round(min(float(element.get("index", 0) or 0) / n_elements, 1.0), 6),
        round(max(0.0, min(y / (viewport * 3.0), 1.0)), 6),
        1.0 if origin == "dom" else 0.0,
        1.0 if origin == "attr" else 0.0,
        1.0 if origin == "text" else 0.0,
        1.0 if origin == "history" else 0.0,
        round(prior_rate, 6),
        round(math.log1p(attempts) / math.log(11.0), 6),
        is_top,
    ]


def heuristic_scores(candidates: Sequence[SelectorCandidate], context: SelectionContext) -> list[float]:
    """Model-free ranking: the explicit fallback when no model file exists.

    Prefers a strong text match on a visible, actionable, shallow selector.
    """
    scores: list[float] = []
    for candidate in candidates:
        features = featurise(candidate, context)
        row = dict(zip(FEATURES, features))
        score = (
            2.0 * row["text_match"]
            + 0.45 * row["prior_success_rate"]
            + 0.25 * row["role_actionable"]
            + 0.20 * row["visible"]
            + 0.10 * row["in_viewport"]
            - 0.30 * row["css_specificity"]
            - 0.10 * row["norm_doc_order"]
        )
        scores.append(round(score, 6))
    return scores


def rank_candidates(
    candidates: Sequence[SelectorCandidate],
    context: SelectionContext,
    predict=None,
) -> list[SelectorCandidate]:
    """Score and sort candidates. `predict(feature_rows) -> scores` if supplied.

    Ties break on the input order, which `build_candidates` already made
    deterministic, so the same inputs always produce the same shortlist.
    """
    items = list(candidates)
    context.n_candidates = len(items)
    feature_rows = [featurise(candidate, context) for candidate in items]
    if predict is not None:
        scores = list(predict(feature_rows) or [])
        if len(scores) != len(items):  # a model that answers partially must not shift rows
            scores = (scores + [0.0] * len(items))[: len(items)]
    else:
        scores = heuristic_scores(items, context)
    enriched = [
        replace(candidate, features=features, score=float(score))
        for candidate, features, score in zip(items, feature_rows, scores)
    ]
    enriched.sort(key=lambda c: (-c.score, ORIGINS.index(c.origin) if c.origin in ORIGINS else 9, c.selector))
    return enriched


__all__ = [
    "FEATURES",
    "ORIGINS",
    "SelectionContext",
    "SelectorCandidate",
    "build_candidates",
    "css_specificity",
    "featurise",
    "feature_names",
    "heuristic_scores",
    "rank_candidates",
    "text_match",
]
