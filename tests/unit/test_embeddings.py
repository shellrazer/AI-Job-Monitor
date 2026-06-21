"""Unit tests for the embeddings module.

Fast tests are fully offline (they never load torch). A single ``slow`` test
loads the real all-MiniLM-L6-v2 model and downloads ~90MB once; it is not
excluded by the default ``addopts`` so it runs on a plain ``pytest``.
"""

from __future__ import annotations

import numpy as np
import pytest

from job_monitor.config import Settings
from job_monitor.embeddings import Embedder, cosine, get_embedder


# --------------------------------------------------------------------------- #
# cosine                                                                      #
# --------------------------------------------------------------------------- #
def test_cosine_orthogonal_is_zero() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(0.0)


def test_cosine_identical_is_one() -> None:
    a = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    assert cosine(a, a) == pytest.approx(1.0)


def test_cosine_opposite_is_minus_one() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine(a, -a) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero() -> None:
    a = np.zeros(4, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert cosine(a, b) == 0.0
    assert cosine(b, a) == 0.0
    assert cosine(a, a) == 0.0


def test_cosine_is_clamped_to_unit_range() -> None:
    # Parallel but non-unit vectors should still land exactly in [-1, 1].
    a = np.array([2.0, 0.0], dtype=np.float32)
    b = np.array([5.0, 0.0], dtype=np.float32)
    val = cosine(a, b)
    assert -1.0 <= val <= 1.0
    assert val == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Embedder laziness                                                           #
# --------------------------------------------------------------------------- #
def test_embedder_does_not_load_model_on_construction() -> None:
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2", model_dir="~/some/dir", dim=384)
    assert emb._model is None
    assert emb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.dim == 384


def test_get_embedder_caches_by_model_name() -> None:
    settings = Settings()
    first = get_embedder(settings)
    second = get_embedder(settings)
    assert first is second
    # Still lazy: building the embedder must not load the model.
    assert first._model is None


# --------------------------------------------------------------------------- #
# fake_embedder fixture sanity check (downstream callable contract)           #
# --------------------------------------------------------------------------- #
def test_fake_embedder_is_deterministic_and_unit_length(fake_embedder) -> None:
    v1 = fake_embedder("National Quality Manager, FSSC 22000")
    v2 = fake_embedder("National Quality Manager, FSSC 22000")
    v3 = fake_embedder("Senior Software QA Engineer")

    assert v1.shape == (384,)
    assert v1.dtype == np.float32
    np.testing.assert_array_equal(v1, v2)  # identical text -> identical vector
    assert not np.array_equal(v1, v3)  # different text -> different vector
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)
    # Downstream code can feed these into cosine().
    assert cosine(v1, v2) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Real model (slow, downloads ~90MB once)                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_model_ranks_food_jd_above_software_jd() -> None:
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    profile = (
        "Senior food-manufacturing quality leader, FSSC 22000, HACCP, multi-site "
        "supplier audits and quality systems."
    )
    food_jd = (
        "National Quality Manager, food manufacturing, FSSC 22000, multi-site, supplier audits"
    )
    software_jd = "Senior Software QA Engineer, Selenium, CI/CD"

    p, food, software = emb.encode([profile, food_jd, software_jd])
    assert p.shape == (384,)
    assert food.shape == (384,)
    assert software.shape == (384,)
    assert cosine(p, food) > cosine(p, software)
