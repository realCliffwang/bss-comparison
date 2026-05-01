"""
Tests for utility functions.
"""

import numpy as np
import pytest
from src.utils import generate_synthetic_mixture, generate_phm_like_cut


class TestGenerateSyntheticMixture:
    """Tests for synthetic mixture generation."""

    def test_basic_generation(self):
        """Test basic mixture generation."""
        S, X, A = generate_synthetic_mixture(
            n_sources=3, n_obs=5, n_samples=1000, fs=1000
        )

        assert S.shape == (3, 1000)
        assert X.shape == (5, 1000)
        assert A.shape == (5, 3)
        assert np.isfinite(S).all()
        assert np.isfinite(X).all()

    def test_mixing_relationship(self):
        """Test that X = A @ S."""
        S, X, A = generate_synthetic_mixture(
            n_sources=3, n_obs=5, n_samples=1000, fs=1000
        )

        X_expected = A @ S
        np.testing.assert_array_almost_equal(X, X_expected)

    def test_vibration_source_type(self):
        """Test vibration source type."""
        S, X, A = generate_synthetic_mixture(
            n_sources=3, n_obs=5, n_samples=1000, fs=1000,
            source_type="vibration"
        )

        assert S.shape[0] == 3

    def test_milling_source_type(self):
        """Test milling source type."""
        S, X, A = generate_synthetic_mixture(
            n_sources=4, n_obs=6, n_samples=1000, fs=1000,
            source_type="milling"
        )

        assert S.shape[0] == 4

    def test_invalid_source_type(self):
        """Test invalid source type raises error."""
        with pytest.raises(ValueError, match="Unknown source_type"):
            generate_synthetic_mixture(source_type="invalid")

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        S1, X1, A1 = generate_synthetic_mixture(seed=42)
        S2, X2, A2 = generate_synthetic_mixture(seed=42)

        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(A1, A2)


class TestGeneratePHMLikeCut:
    """Tests for PHM-like cut generation."""

    def test_basic_generation(self):
        """Test basic PHM cut generation."""
        signals, fs, wear = generate_phm_like_cut(
            tool_id=1, cut_no=100, n_samples=5000
        )

        assert signals.shape == (7, 5000)
        assert fs == 50000
        assert isinstance(wear, float)

    def test_wear_increases(self):
        """Test that wear increases with cut number."""
        _, _, wear_early = generate_phm_like_cut(cut_no=10)
        _, _, wear_late = generate_phm_like_cut(cut_no=300)

        assert wear_late > wear_early

    def test_different_seeds(self):
        """Test different seeds produce different results."""
        s1, _, _ = generate_phm_like_cut(cut_no=1, seed=1)
        s2, _, _ = generate_phm_like_cut(cut_no=1, seed=2)

        assert not np.array_equal(s1, s2)
