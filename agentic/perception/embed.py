"""Deterministic text embedding for page snapshots.

Two implementations behind one interface:

`TfidfSvdEmbedder`
    TF-IDF over word unigrams (+ bigrams where the corpus allows) hstacked with
    character n-grams in word boundaries, reduced by TruncatedSVD and row
    L2-normalised. The char channel is what makes this survive real DOMs:
    "Submit", "submit-btn" and "submitted" stay close even though a word-only
    model sees three disjoint tokens. Character n-grams also cover OOV, which
    matters because every site invents its own class names.

`HashingEmbedder`
    Signed feature hashing with blake2b into a fixed-width vector. No corpus, no
    fitting, no scikit-learn — used as the fallback whenever the SVD path cannot
    be built (one-element snapshot, degenerate vocabulary, scikit-learn absent).

Both are *deterministic*: the same inputs always produce the same vectors, bit
for bit. TruncatedSVD's randomised solver is pinned to `random_state=0` for
exactly that reason. That is a hard requirement here because the agent's
behaviour is fed into assertions in tests and into an episodic memory whose
hashes must remain comparable across runs.

This is a lexical-semantic embedder, not a neural one. It captures surface form
and morphology, not paraphrase ("log in" vs "authenticate" are far apart unless
they share characters). See the limitations section in README.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import numpy as np

try:  # scikit-learn is an install dependency, but the fallback must work anyway
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without sklearn
    TruncatedSVD = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise, leaving all-zero rows at zero instead of NaN."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 1e-12)
    return out.astype(np.float32, copy=False)


class HashingEmbedder:
    """Signed feature hashing. Stateless, so `fit` is a no-op."""

    name = "hashing-blake2b"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)
        self._fitted = True

    def _bucket(self, token: str) -> tuple[int, float]:
        # Two independent digests: one picks the bucket, one picks the sign.
        # Separate hashing keeps the sign from correlating with the index.
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        lowered = str(text or "").lower()
        for word in _word_split(lowered):
            yield word
            if len(word) > 3:
                yield "#" + word[:3]
                yield word[:3] + "#"

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            tokens = list(self._tokens(text))
            total = len(tokens) or 1
            for token in tokens:
                index, sign = self._bucket(token)
                counts[index] = counts.get(index, 0.0) + sign / math.log(2.0 + total)
            for index, value in counts.items():
                out[row, index] = value
        return l2_normalise(out)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform(texts)


def _word_split(text: str) -> list[str]:
    split: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            split.append("".join(current))
            current = []
    if current:
        split.append("".join(current))
    return split


class TfidfSvdEmbedder:
    """Word + char TF-IDF, hstacked, reduced by TruncatedSVD, L2-normalised."""

    name = "tfidf-word+char_wb-svd"

    def __init__(self, dim: int = 96, char_ngram: tuple[int, int] = (3, 5), min_df: int = 1):
        self.dim = int(dim)
        self.char_ngram = tuple(char_ngram)
        self.min_df = int(min_df)
        self._word = None
        self._char = None
        self._svd = None
        self._fallback: HashingEmbedder | None = None

    @property
    def fallback_active(self) -> bool:
        return self._fallback is not None

    def _vectorizers(self):
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=self.min_df,
            lowercase=True,
            token_pattern=r"(?u)\b[\w:-]+\b",
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram,
            sublinear_tf=True,
            min_df=1,
            lowercase=True,
        )
        return word, char

    def fit(self, texts: Sequence[str]) -> "TfidfSvdEmbedder":
        corpus = [str(t or "") for t in texts]
        self._fallback = None
        if not SKLEARN_AVAILABLE or not any(c.strip() for c in corpus):
            self._fallback = HashingEmbedder(dim=max(self.dim, 64))
            return self
        self._word, self._char = self._vectorizers()
        try:
            from scipy.sparse import hstack

            w = self._word.fit_transform(corpus)
            c = self._char.fit_transform(corpus)
            merged = hstack([w, c])
            n_docs, n_features = merged.shape
            # TruncatedSVD needs n_components < n_features and >= 1 sample.
            components = max(1, min(self.dim, n_features - 1, n_docs))
            if n_features < 2:
                raise ValueError("vocabulary too small for a matrix factorisation")
            self._svd = TruncatedSVD(n_components=components, random_state=0, algorithm="randomized")
            self._svd.fit(merged)
        except Exception:  # noqa: BLE001 - any degeneracy lands on the safe path
            self._word = self._char = self._svd = None
            self._fallback = HashingEmbedder(dim=max(self.dim, 64))
            self._fallback.fit(corpus)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        corpus = [str(t or "") for t in texts]
        if self._fallback is not None:
            return self._fallback.transform(corpus)
        if self._svd is None:
            self.fit(corpus)
            return self.transform(corpus)
        from scipy.sparse import hstack

        merged = hstack([self._word.transform(corpus), self._char.transform(corpus)])
        # A held-out document can miss the fitted vocabulary entirely; SVD maps
        # the all-zero row to zero, and the caller treats zero norm as "no match".
        return l2_normalise(self._svd.transform(merged))

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


CosineEmbedder = TfidfSvdEmbedder


def make_embedder(dim: int = 96, prefer: str = "tfidf-svd") -> TfidfSvdEmbedder | HashingEmbedder:
    """Embedder factory. `prefer="hashing"` or an unavailable sklearn gives the
    corpus-free hashing path, which is also useful for A/B-ing recall quality."""
    if prefer == "hashing" or not SKLEARN_AVAILABLE:
        return HashingEmbedder(dim=max(dim, 64))
    return TfidfSvdEmbedder(dim=dim)


__all__ = [
    "HashingEmbedder",
    "TfidfSvdEmbedder",
    "CosineEmbedder",
    "make_embedder",
    "l2_normalise",
    "SKLEARN_AVAILABLE",
]
