"""Synthetic-but-labelled fixture corpus for the selector ranker.

There is no public dataset of "which CSS selector would have worked for this
goal on this page", and building a real one means crawling and hand-annotating.
This module generates a corpus whose *structure* is realistic — goals, one
correct widget, and distractors that fail for the specific reasons real
selectors fail — so the learned ranking function can be validated and the
persistence path exercised in CI without a browser.

What this does **not** claim: that the reported numbers transfer to the live
web. The generator's causal structure is written down in
`docs/agent-perception.md`, the candidate order is document order (the naive
baseline), and every group is reproducible from its seed. Treat the metrics as
"the learner recovers a known signal from the features we expose", not as
"this is what it will do on your site".

Three separable axes, deliberately sampled independently of each other:

* **identity** — which widget the element is. Exactly one element per group is
  the right one for the goal; the rest are lexically adjacent distractors.
* **way** — why a distractor is wrong: zero-size, below the fold, a wrapper with
  no behaviour, an id from a dead render, or a collapsed-menu *clone* of the
  right widget (same text, unclickable).
* **shape** — what address the extraction script reported: a brittle
  `nth-of-type` chain, a bare `#id`, or an id-plus-class form.

Independence is the point. If brittle paths were only ever handed to distractors
(the obvious way to write this generator, and the way an earlier revision of it
was written), `css_specificity` would be a proxy for "is a distractor" and the
negative weight we want the learner to discover would be an artefact of the
fixture instead of a lesson about the web.

The supervision signal is `GeneratedGroup.label_for`: right widget **and** a
live address. A brittle path on the correct widget is labelled 0 exactly when a
banner was inserted above the fold, which is the one situation where brittleness
actually bites — so specificity carries cost only in the presence of a text
match strong enough to otherwise win.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Sequence

from ..snapshot import element_text
from .feedback import FeedbackRecord, FeedbackStore, make_group_id
from .selectors import FEATURES, SelectionContext, SelectorCandidate, featurise

#: Goal families: (goal text, correct label, actionable role)
_GOAL_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("sign in to my account", "Sign in", "button"),
    ("create a new account", "Create account", "button"),
    ("search for laptops", "Search products", "searchbox"),
    ("filter results by price", "Price range", "combobox"),
    ("add the item to the cart", "Add to cart", "button"),
    ("unsubscribe from the newsletter", "Unsubscribe", "link"),
    ("upload a profile photo", "Choose file", "input"),
    ("reset my password", "Reset password", "link"),
    ("apply the coupon code", "Apply coupon", "button"),
    ("change the delivery address", "Edit address", "button"),
    ("subscribe to the newsletter", "Subscribe", "button"),
    ("open the support chat", "Chat with support", "button"),
)

#: Distractor widget labels: lexically adjacent, semantically wrong.
_DISTRACTORS: tuple[tuple[str, str], ...] = (
    ("Sign out", "button"),
    ("Settings", "link"),
    ("Shopping cart", "link"),
    ("All categories", "combobox"),
    ("Sort by relevance", "combobox"),
    ("Next page", "link"),
    ("Previous", "button"),
    ("Home", "link"),
    ("Privacy policy", "link"),
    ("Accept cookies", "button"),
    ("Dismiss", "button"),
    ("Menu", "button"),
)

#: How a *distractor* is wrong, mirroring real selector failures. Each entry is
#: (name, sampling weight); the effect of each is spelled out in `_element`.
#:
#: hidden          zero-size node (`display: none`) the driver cannot click
#: offscreen       real widget, below the fold, and this step never scrolled
#: non-actionable  a generic wrapper: no role, no behaviour, no id
#: stale-id        id carried over from a previous render, now a dead address
#: near-match      collapsed mobile-menu clone of the *correct* widget: the same
#:                 text, so a text-matching ranker cannot separate them
_WAYS: tuple[tuple[str, float], ...] = (
    ("hidden", 0.24),
    ("offscreen", 0.20),
    ("non-actionable", 0.20),
    ("stale-id", 0.16),
    ("near-match", 0.20),
)

#: What address the extraction script reported for a node. Sampled for the
#: correct widget too, and from the same distribution.
#:
#: path        `body > div:nth-of-type(4) > ... > span:nth-of-type(7)`
#: id          `#e7`
#: qualified   `#e7.alt-variant`
_SHAPES: tuple[tuple[str, float], ...] = (("path", 0.30), ("id", 0.52), ("qualified", 0.18))

#: Chance that the *right* widget itself is unaddressable this render (dead id
#: and no live variant). Those groups become dead ends with no positive row,
#: which `ranking_metrics` skips rather than scoring as a miss.
_DEAD_END_CHANCE = 0.06

#: Perception noise. The features the ranker sees are *measurements taken from a
#: live page*, not ground truth: the extraction script reads `offsetWidth` while a
#: transition is running, a component library ships a button with no ARIA role, a
#: decorative `<div>` carries `role="button"` and does nothing. Modelling that is
#: the difference between a fixture a ranker scores 1.000 on and one that has an
#: error floor. `_element` therefore stores the truth under `_clickable` and lets
#: the reported `visible` / `role` / `in_viewport` disagree with it.
_NOISE_VISIBLE = 0.10
_NOISE_ROLE_DOWNGRADE = 0.10
_NOISE_ROLE_FAKE = 0.08
_NOISE_VIEWPORT = 0.10

_TAGS = {"button": "button", "link": "a", "searchbox": "input", "combobox": "select", "input": "input"}


@dataclass
class GeneratedGroup:
    goal: str
    state_hash: str
    step: int
    action: str
    correct_index: int
    correct_selector: str
    candidates: list[SelectorCandidate]
    context: "SelectionContext" = field(default_factory=SelectionContext)
    #: A banner was inserted above the fold, so every brittle `nth-of-type` path
    #: the DOM handed us now resolves to the wrong node.
    layout_shifted: bool = False

    def is_correct(self, candidate: SelectorCandidate) -> bool:
        """Does this selector address the right *widget*, whatever its odds?

        Deliberately wider than `label_for`: an element's `#id` and its deep DOM
        path both point at the same widget. Relevance is only half the question.
        """
        return int(candidate.element.get("index", -1)) == self.correct_index

    def resolves(self, candidate: SelectorCandidate) -> bool:
        """Would the driver actually have acted on this address, right now?

        Three real failure modes: a zero-size node, a positional path invalidated
        by the layout shift, an id carried over from a previous render. The first
        is read from `_clickable` (generator ground truth), *not* from the
        reported `visible` flag the ranker is featurised on — the two disagree on
        purpose, and that disagreement is the error floor on `top1_accuracy`.
        """
        element = candidate.element or {}
        if not element.get("_clickable", element.get("visible", True)):
            return False
        selector = candidate.selector
        if "-legacy-" in selector:
            return False
        if self.layout_shifted and candidate.origin == "dom" and ":nth-of-type(" in selector:
            return False
        return True

    def label_for(self, candidate: SelectorCandidate) -> int:
        """The supervision signal: right widget *and* a live address."""
        return 1 if (self.is_correct(candidate) and self.resolves(candidate)) else 0

    @property
    def is_dead_end(self) -> bool:
        return not self.correct_selector


def _sample(rng: random.Random, weighted: tuple[tuple[str, float], ...]) -> str:
    return rng.choices([name for name, _ in weighted], weights=[w for _, w in weighted], k=1)[0]


def _dom_selector(rng: random.Random, index: int, element_id: str, shape: str) -> str:
    """The address the extraction script reported for this node.

    `element_id` may already carry the `-legacy-` token; a wrapper without an id
    gets a class-ish path instead, because there is no `#` to write.
    """
    if shape == "path":
        return (
            f"body > div:nth-of-type({rng.randint(2, 9)}) > section > "
            f"ul:nth-of-type({rng.randint(1, 4)}) > li > span:nth-of-type({index + 1})"
        )
    if not element_id:
        return f"div.promo-banner-{index} > span"
    if shape == "qualified":
        return f"#{element_id}.{rng.choice(['nav', 'muted', 'alt'])}-variant"
    return f"#{element_id}"


def _element(rng: random.Random, label: str, role: str, index: int, way: str, shape: str) -> dict:
    """One node in the snapshot: `way` says what is wrong with it, if anything.

    The dict carries both layers. `_clickable` / `_actionable` are the generator's
    ground truth and are read only when labelling a row. `visible`, `in_viewport`
    and `role` are what an extraction script would have *reported*, and they lie
    some of the time, because a ranker featurised on truth would score perfectly
    and tell us nothing.
    """
    actionable = way != "non-actionable"
    clickable = way not in ("hidden", "near-match")
    offscreen = way in ("offscreen", "near-match")
    # Distinct per element, even for a wrapper: duplicate selectors are dropped
    # in `generate_group`, and a dropped row would shift the position of the
    # widget the group labels as correct.
    element_id = f"e{index}" if actionable else ""
    if way == "stale-id":
        # The node really does report this id in *this* snapshot; it just no
        # longer exists to click, because the id was minted by a dead render.
        element_id = f"e{index}-legacy-{rng.randint(100, 999)}"
    visible = clickable if rng.random() >= _NOISE_VISIBLE else not clickable
    viewport_truth = clickable and not offscreen
    in_viewport = viewport_truth if rng.random() >= _NOISE_VIEWPORT else not viewport_truth
    if not actionable:
        # An inert wrapper that advertises itself as a button: the role feature is
        # exactly as trustworthy as the markup it came from.
        reported_role = "button" if rng.random() < _NOISE_ROLE_FAKE else "generic"
    elif rng.random() < _NOISE_ROLE_DOWNGRADE:
        reported_role = "generic"  # real control, no ARIA role written
    else:
        reported_role = role
    return {
        "selector": _dom_selector(rng, index, element_id, shape),
        "label": label,
        "role": reported_role,
        "tag": (_TAGS.get(role, "span") if actionable else "div"),
        "attrs": {"id": element_id},
        "rect": {
            "x": float(rng.randint(0, 400)),
            "y": float(rng.randint(1200, 2400) if offscreen else rng.randint(-2400, 800)),
            "width": float(rng.randint(80, 220)) if visible else 0.0,
            "height": float(rng.randint(24, 44)) if visible else 0.0,
        },
        "visible": visible,
        "in_viewport": in_viewport,
        "index": index,
        "confidence": 0.0,
        "way": way,
        "_clickable": clickable,
        "_actionable": actionable,
    }


def _state_hash(seed: int, group_index: int) -> str:
    """A digest, not an interpolated string.

    `make_group_id` keeps the first 12 characters of the state hash, so a
    readable-but-truncating value used to collapse ~10 generated groups into one
    ranking group and quietly merge their shortlists in the metrics.
    """
    return hashlib.blake2b(f"{seed}:{group_index}".encode("utf-8"), digest_size=16).hexdigest()


def generate_group(seed: int, group_index: int = 0) -> GeneratedGroup:
    """One labelled shortlist. Deterministic in `(seed, group_index)`."""
    rng = random.Random(f"{seed}:{group_index}")
    goal, correct_label, correct_role = _GOAL_TEMPLATES[group_index % len(_GOAL_TEMPLATES)]
    layout_shifted = rng.random() < 0.35
    n_distractors = rng.randint(3, 6)
    pool = [d for d in _DISTRACTORS if d[0] != correct_label]
    rng.shuffle(pool)

    correct_index = rng.randrange(n_distractors + 1)  # position in document order
    elements: list[dict] = []
    for position in range(n_distractors + 1):
        shape = _sample(rng, _SHAPES)
        if position == correct_index:
            # The right widget is never hidden, never a wrapper and never below
            # the fold - but it can be addressed only by a brittle path, and it
            # can be unaddressable outright.
            way = "stale-id" if rng.random() < _DEAD_END_CHANCE else ""
            label, role = correct_label, correct_role
        else:
            way = _sample(rng, _WAYS)
            if way == "near-match":
                label, role = correct_label, correct_role  # same text, dead menu clone
            else:
                label, role = pool[(position + group_index) % len(pool)]
        elements.append(_element(rng, label, role, position, way, shape))

    context = SelectionContext(
        goal=goal, observation="", action="click", n_elements=len(elements), viewport_height=900
    )
    candidates: list[SelectorCandidate] = []
    seen: set[str] = set()

    def add(element: dict, selector: str, origin: str) -> None:
        if not selector or selector in seen:
            return  # a cleanly-addressed node's DOM path *is* its `#id`
        seen.add(selector)
        candidates.append(SelectorCandidate(element=element, selector=selector, origin=origin, text=element_text(element)))

    for element in elements:
        # Document order, not relevance order: that is the naive baseline.
        add(element, str(element["selector"]), "dom")
    for element in elements:
        attrs = element.get("attrs") or {}
        reported = str(attrs.get("id") or "")
        if reported.startswith("e"):
            add(element, f"#{reported}", "attr")
    context.n_candidates = len(candidates)
    for candidate in candidates:
        candidate.features = featurise(candidate, context)

    group = GeneratedGroup(
        goal=goal,
        state_hash=_state_hash(seed, group_index),
        step=group_index % 8 + 1,
        action="click",
        correct_index=correct_index,
        correct_selector="",
        candidates=candidates,
        context=context,
        layout_shifted=layout_shifted,
    )
    # What the agent would have shipped: the first live address of the right
    # widget, in offer order. Empty means "no live address in this snapshot".
    for candidate in candidates:
        if group.label_for(candidate) == 1:
            group.correct_selector = candidate.selector
            break
    return group


def generate_records(groups: Sequence[int] | int = 240, seed: int = 17) -> list[FeedbackRecord]:
    """Flatten generated groups into labelled feedback rows.

    `groups` is either a count (indices `0..n-1`) or the explicit group indices
    to build. Everything is reproducible from `(seed, index)`.
    """
    indices = range(int(groups)) if isinstance(groups, int) else [int(g) for g in groups]
    records: list[FeedbackRecord] = []
    for index in indices:
        group = generate_group(seed=seed, group_index=index)
        group_id = make_group_id(group.state_hash, group.goal, group.step, group.action)
        for rank, candidate in enumerate(group.candidates):
            label = group.label_for(candidate)
            records.append(
                FeedbackRecord(
                    group_id=group_id,
                    state_hash=group.state_hash,
                    goal=group.goal,
                    action=group.action,
                    selector=candidate.selector,
                    origin=candidate.origin,
                    role=str(candidate.element.get("role", "unknown")),
                    offered_rank=rank,
                    chosen=candidate.selector == group.correct_selector,
                    label=label,
                    outcome="ok" if label == 1 else "failed",
                    features=list(candidate.features),
                )
            )
    return records


def corpus_stats(groups: Sequence[int] | int = 240, seed: int = 17) -> dict[str, float]:
    """Structure of the fixture, so a doc claim can be checked instead of trusted."""
    indices = list(range(int(groups))) if isinstance(groups, int) else [int(g) for g in groups]
    built = [generate_group(seed=seed, group_index=i) for i in indices]
    n_groups = max(len(built), 1)
    dead_ends = sum(1 for g in built if g.is_dead_end)
    scored = [g for g in built if not g.is_dead_end]
    n_scored = max(len(scored), 1)
    positive_rows = candidates = 0
    for group in built:
        candidates += len(group.candidates)
        positive_rows += sum(1 for c in group.candidates if group.label_for(c) == 1)
    brittle = 0
    visible_errors = role_errors = nodes = 0
    for group in built:
        nodes += len(group.candidates)
        for candidate in group.candidates:
            element = candidate.element
            reports_generic = str(element.get("role")) == "generic"
            if bool(element.get("visible")) is not bool(element.get("_clickable")):
                visible_errors += 1
            if reports_generic is bool(element.get("_actionable", True)):
                role_errors += 1  # "generic" on a real control, or a role on a wrapper
        if not group.is_dead_end and any(
            c.origin == "dom" and group.is_correct(c) and ":nth-of-type(" in c.selector for c in group.candidates
        ):
            brittle += 1
    rows = max(nodes, 1)
    return {
        "groups": len(built),
        "candidate_rows": nodes,
        "avg_candidates": round(nodes / n_groups, 2),
        "positive_rows": positive_rows,
        "positive_rate": round(positive_rows / max(candidates, 1), 4),
        "dead_end_groups": dead_ends,
        "layout_shifted_rate": round(sum(1 for g in built if g.layout_shifted) / n_groups, 4),
        "correct_widget_is_doc_first_rate": round(sum(1 for g in scored if g.correct_index == 0) / n_scored, 4),
        "brittle_address_on_correct_rate": round(brittle / n_scored, 4),
        "reported_visible_error_rate": round(visible_errors / rows, 4),
        "reported_role_error_rate": round(role_errors / rows, 4),
    }


def build_store(n_groups: int = 240, path: str | None = ":memory:", seed: int = 17) -> FeedbackStore:
    """A FeedbackStore populated with the synthetic corpus (idempotent)."""
    store = FeedbackStore(path=path)
    if len(store) == 0:
        store.extend(generate_records(range(n_groups), seed=seed))
    return store


FEATURE_COUNT = len(FEATURES)

__all__ = [
    "FEATURE_COUNT",
    "GeneratedGroup",
    "build_store",
    "corpus_stats",
    "generate_group",
    "generate_records",
]
