"""Perception: turning a page snapshot into a small, relevant element set.

Layers, lowest first:

- `snapshot` (top-level module) — normalised element dicts + structure hashes.
- `embed`    — deterministic TF-IDF(word)+char_wb → TruncatedSVD embedder, with
               a signed blake2b hashing fallback that needs no scikit-learn.
- `index`    — in-memory vector index over the snapshot + MMR re-ranking.
- `memory`   — SQLite episodic store of visited states (loop breaking + recall).
- `selectors`— candidate selector generation and interpretable featurisation.
- `ranker`   — scikit-learn model that ranks candidates before the LLM sees them.
- `feedback` — append-only log of (candidate, features, outcome) for retraining.
"""
from __future__ import annotations

from .corpus import build_store as build_synthetic_store
from .corpus import generate_group, generate_records
from .embed import CosineEmbedder, HashingEmbedder, TfidfSvdEmbedder, make_embedder
from .feedback import FeedbackRecord, FeedbackStore
from .index import ElementIndex, Selection, SelectionHit, select_elements
from .memory import Episode, EpisodicMemory, LoopWarning, StateSignature
from .ranker import RankerMetrics, SelectorRanker, ranking_metrics, split_rows
from .selectors import SelectorCandidate, SelectionContext, build_candidates, feature_names, featurise

__all__ = [
    "CosineEmbedder",
    "ElementIndex",
    "Episode",
    "FeedbackRecord",
    "FeedbackStore",
    "HashingEmbedder",
    "LoopWarning",
    "RankerMetrics",
    "Selection",
    "SelectionContext",
    "SelectionHit",
    "SelectorCandidate",
    "SelectorRanker",
    "StateSignature",
    "TfidfSvdEmbedder",
    "build_candidates",
    "build_synthetic_store",
    "feature_names",
    "featurise",
    "generate_group",
    "generate_records",
    "make_embedder",
    "ranking_metrics",
    "select_elements",
]
