"""Sentence-embedding helpers (spec §7).

Wraps a sentence-transformers model behind a lazily-loaded :class:`Embedder`
so importing this module (and constructing an ``Embedder``) never pulls in
torch — the model is only loaded on the first :meth:`Embedder.encode` call.
Also provides a numerically safe :func:`cosine` and a module-level cache keyed
by model name via :func:`get_embedder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from job_monitor.config import expand_path

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from job_monitor.config import Settings


class Embedder:
    """Lazy wrapper around a sentence-transformers model.

    The underlying model is loaded on first use (see :meth:`encode`), not in
    ``__init__`` and not at import time, so unit tests can construct an
    ``Embedder`` without ever importing torch.
    """

    def __init__(self, model_name: str, model_dir: str | None = None, dim: int = 384) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.dim = dim
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        """Load and cache the model on first access (imports torch here)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self.model_dir is not None:
                kwargs["cache_folder"] = str(expand_path(self.model_dir))
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text into unit-length float32 vectors.

        Returns shape ``(dim,)`` for a single string and ``(n, dim)`` for a
        list of strings. Vectors are L2-normalized (``normalize_embeddings``).
        """
        model = self._load()
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        vectors = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors[0] if single else vectors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]``; zero vectors yield ``0.0``.

    Pure numpy and numerically safe: clamps the result to the valid range to
    absorb floating-point overshoot from already-normalized inputs.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    return float(np.clip(sim, -1.0, 1.0))


_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(settings: Settings) -> Embedder:
    """Build (or reuse) an :class:`Embedder` from ``settings.embeddings``.

    Cached module-level by ``model_name`` so repeated calls share one instance
    (and therefore one loaded model).
    """
    cfg = settings.embeddings
    cached = _EMBEDDER_CACHE.get(cfg.model_name)
    if cached is None:
        cached = Embedder(model_name=cfg.model_name, model_dir=cfg.model_dir, dim=cfg.dim)
        _EMBEDDER_CACHE[cfg.model_name] = cached
    return cached


__all__ = ["Embedder", "cosine", "get_embedder"]
