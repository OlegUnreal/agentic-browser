"""Vector index over a page snapshot, with MMR re-ranking.

Why re-rank at all: cosine ranking over a DOM returns ten near-identical
widgets. A product page has "Add to cart" in a sticky header, next to each of
twelve cards, and in a toast — all of them score within a few thousandths of the
query, so the model is handed ten copies of the same button and never sees the
one "Add to cart" that is actually in the checkout form it needs.

Maximal Marginal Relevance buys diversity explicitly:

    MMR(d) = lambda * sim(d, q) - (1 - lambda) * max(sim(d, s) for s in chosen)

`lambda = 1.0` degenerates to pure cosine. The default 0.72 keeps relevance on
top while pruning near-duplicates, chosen so that on the synthetic snapshot in
`tests/test_retrieval.py` the greedy pick is the same as the oracle for the
easy cases and diversity only kicks in where it helps.

Relevance alone is not enough either: a widget the user can actually see and
click gets a small deterministic prior boost (`visibility`, `in_viewport`,
document order), because the agent's actions on off-screen nodes fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..snapshot import clean_label, coerce_elements, element_text
from .embed import HashingEmbedder, TfidfSvdEmbedder, make_embedder

#: Weight of the diversity term in MMR. 1.0 == plain cosine ranking.
DEFAULT_LAMBDA_MMR = 0.72

#: Additive prior weights: a hidden element is one the driver will fail on.
_ROLE_PRIOR = {
    "button": 0.030,
    "link": 0.025,
    "submit": 0.030,
    "textbox": 0.020,
    "searchbox": 0.020,
    "combobox": 0.015,
    "checkbox": 0.010,
    "radio": 0.010,
    "menuitem": 0.010,
    "generic": 0.0,
    "unknown": -0.010,
}


@dataclass
class SelectionHit:
    """One retrieved element with the scores that produced it."""

    element: dict
    rank: int
    relevance: float
    diversity: float
    score: float
    mmr_rank: int = 0
    text: str = ""

    @property
    def selector(self) -> str:
        return str(self.element.get("selector", ""))

    @property
    def label(self) -> str:
        return str(self.element.get("label", ""))

    @property
    def role(self) -> str:
        return str(self.element.get("role", "unknown"))

    def as_dict(self) -> dict:
        """Prompt-friendly view: what the model needs, nothing it does not."""
        return {
            "selector": self.selector,
            "label": self.label,
            "role": self.role,
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 4),
            "in_viewport": bool(self.element.get("in_viewport", True)),
        }


@dataclass
class Selection:
    """Result of one retrieval: selected elements plus diagnostics."""

    hits: list[SelectionHit] = field(default_factory=list)
    query: str = ""
    corpus_size: int = 0
    embedder: str = ""
    candidate_order: list[int] = field(default_factory=list)
    #: Index into the *corpus* (`ElementIndex.elements`) of the best purely
    #: relevant pick; lets tests compare MMR against cosine without a second pass.
    greedy_best_index: int = -1

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)

    @property
    def selectors(self) -> list[str]:
        return [h.selector for h in self.hits]

    @property
    def labels(self) -> list[str]:
        return [h.label for h in self.hits]

    def as_dicts(self) -> list[dict]:
        return [h.as_dict() for h in self.hits]

    @property
    def top(self) -> SelectionHit | None:
        return self.hits[0] if self.hits else None


def _role_prior(element: dict) -> float:
    role = str(element.get("role", "unknown"))
    prior = _ROLE_PRIOR.get(role, 0.0)
    if not element.get("visible", True):
        prior -= 0.05
    if element.get("in_viewport", True) is False:
        prior -= 0.02
    return prior


def mmr_select(
    matrix: np.ndarray,
    query: np.ndarray,
    order: Sequence[int],
    k: int,
    lambda_mm: float = DEFAULT_LAMBDA_MMR,
) -> list[tuple[int, float, float]]:
    """Greedy MMR over `order` (already sorted by descending relevance).

    Returns [(index, relevance, diversity_penalty)]. Greedy is exact-enough at
    snapshot sizes (tens to a few hundred nodes) and keeps the whole loop
    deterministic and debuggable.
    """
    remaining = list(order)
    chosen: list[tuple[int, float, float]] = []
    selected_vectors: list[np.ndarray] = []
    while remaining and len(chosen) < k:
        best_index, best_score, best_relevance, best_penalty = remaining[0], -np.inf, 0.0, 0.0
        for index in remaining:
            relevance = float(matrix[index] @ query)
            if selected_vectors:
                penalty = max(float(matrix[index] @ previous) for previous in selected_vectors)
            else:
                penalty = 0.0
            score = lambda_mm * relevance - (1.0 - lambda_mm) * penalty
            if score > best_score + 1e-12:
                best_index, best_score, best_relevance, best_penalty = index, score, relevance, penalty
        chosen.append((best_index, best_relevance, best_penalty))
        selected_vectors.append(matrix[best_index])
        remaining.remove(best_index)
    return chosen


class ElementIndex:
    """Embed + retrieve the interactive elements of a single snapshot.

    Scope is one page state, not the whole crawl: the vocabulary is fitted on
    the snapshot itself plus the query, which is what keeps embeddings stable
    and comparable run to run and avoids a global model that drifts as the
    agent browses.
    """

    def __init__(
        self,
        elements: Sequence[dict] | None,
        embedder: TfidfSvdEmbedder | HashingEmbedder | None = None,
        dim: int = 96,
        prefer: str = "tfidf-svd",
    ):
        self.elements: list[dict] = coerce_elements(elements)
        self._texts = [element_text(e) for e in self.elements]
        self.embedder = embedder if embedder is not None else make_embedder(dim=dim, prefer=prefer)
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self._texts:
                self._matrix = np.zeros((0, 1), dtype=np.float32)
            else:
                self._matrix = self.embedder.fit_transform(self._texts)
        return self._matrix

    def retrieve(
        self,
        query: str,
        k: int = 8,
        lambda_mm: float = DEFAULT_LAMBDA_MMR,
        prior: str | None = None,
    ) -> Selection:
        """Top-k elements for `query`, diversified by MMR.

        `prior` is optional extra text (usually the last observation: the action
        that just ran and what it produced) appended to the goal so the
        retrieval follows the dialogue instead of the original instruction only.
        """
        query = clean_label(query, 400)
        combined = f"{query} {clean_label(prior or '', 200)}".strip()
        if not self.elements or not combined:
            return Selection(query=combined, corpus_size=len(self.elements), embedder=self.embedder.name)
        matrix = self.matrix
        qvec = self.embedder.transform([combined])[0]
        if float(np.linalg.norm(qvec)) < 1e-9:
            return Selection(query=combined, corpus_size=len(self.elements), embedder=self.embedder.name)
        similarities = matrix @ qvec
        priors = np.array([_role_prior(e) for e in self.elements], dtype=np.float64)
        # Document order breaks score ties so retrieval never depends on float noise.
        keyed = sorted(
            range(len(self.elements)),
            key=lambda i: (-(float(similarities[i]) + priors[i]), i),
        )
        greedy_best = keyed[0]
        cosine_rank = {index: position for position, index in enumerate(keyed)}
        picked = mmr_select(matrix, qvec, keyed, k=k, lambda_mm=lambda_mm)
        hits: list[SelectionHit] = []
        for rank, (index, relevance, penalty) in enumerate(picked):
            score = float(similarities[index]) + float(priors[index])
            hits.append(
                SelectionHit(
                    element=self.elements[index],
                    rank=rank,
                    relevance=relevance,
                    diversity=penalty,
                    score=score,
                    mmr_rank=int(cosine_rank.get(index, -1)),
                    text=self._texts[index],
                )
            )
        return Selection(
            hits=hits,
            query=combined,
            corpus_size=len(self.elements),
            embedder=self.embedder.name,
            candidate_order=keyed,
            greedy_best_index=greedy_best,
        )


def select_elements(
    elements: Sequence[dict] | None,
    query: str,
    k: int = 8,
    lambda_mm: float = DEFAULT_LAMBDA_MMR,
    prior: str | None = None,
    embedder: TfidfSvdEmbedder | HashingEmbedder | None = None,
) -> Selection:
    """One-shot convenience wrapper around `ElementIndex`."""
    index = ElementIndex(elements, embedder=embedder)
    return index.retrieve(query, k=k, lambda_mm=lambda_mm, prior=prior)


__all__ = ["ElementIndex", "Selection", "SelectionHit", "select_elements", "mmr_select", "DEFAULT_LAMBDA_MMR"]
