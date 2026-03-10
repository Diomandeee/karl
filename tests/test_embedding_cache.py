"""Tests for the embedding cache."""

import pytest
from karl.embedding_cache import (
    cache_key,
    cosine_similarity,
    rank_skills,
)


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("hello world")
        k2 = cache_key("hello world")
        assert k1 == k2

    def test_different_inputs(self):
        k1 = cache_key("hello")
        k2 = cache_key("world")
        assert k1 != k2

    def test_length(self):
        k = cache_key("test input")
        assert len(k) == 16


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity([1, 2], [1, 2, 3]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


class TestRankSkills:
    def test_basic_ranking(self):
        prompt_vec = [1.0, 0.0, 0.0]
        skills = {
            "deploy": ([1.0, 0.0, 0.0], 1.0),      # Perfect match
            "debug": ([0.0, 1.0, 0.0], 1.0),        # Orthogonal
            "build": ([0.7, 0.7, 0.0], 1.0),        # Partial match
        }
        ranked = rank_skills(prompt_vec, skills, threshold=0.3)
        assert len(ranked) >= 1
        assert ranked[0][0] == "deploy"  # Highest similarity

    def test_weight_affects_ranking(self):
        prompt_vec = [1.0, 0.0, 0.0]
        skills = {
            "a": ([0.8, 0.6, 0.0], 0.5),   # High sim, low weight
            "b": ([0.7, 0.7, 0.0], 1.5),    # Lower sim, high weight
        }
        ranked = rank_skills(prompt_vec, skills, threshold=0.3)
        # b should rank higher due to weight boost
        if len(ranked) == 2:
            assert ranked[0][0] == "b"

    def test_threshold_filtering(self):
        prompt_vec = [1.0, 0.0, 0.0]
        skills = {
            "weak": ([0.1, 0.9, 0.0], 1.0),  # Low similarity
        }
        ranked = rank_skills(prompt_vec, skills, threshold=0.5)
        assert len(ranked) == 0

    def test_empty_skills(self):
        ranked = rank_skills([1.0, 0.0], {})
        assert ranked == []
