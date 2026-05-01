"""
Tests for BSS module.
"""

import numpy as np
import pytest
from src.bss_module import (
    run_sobi,
    run_fastica,
    run_jade,
    run_nmf,
    run_pca,
    run_bss,
    bss_factory,
    _joint_diagonalize_jacobi,
)


class TestSOBI:
    """Tests for SOBI algorithm."""

    def test_basic_sobi(self, sample_bss_mixture):
        """Test basic SOBI separation."""
        S_true, X, A, fs = sample_bss_mixture
        n_sources = S_true.shape[0]

        S_est, W = run_sobi(X, n_sources=n_sources, n_lags=10)

        assert S_est.shape[0] == n_sources
        assert S_est.shape[1] == X.shape[1]
        assert W.shape == (n_sources, X.shape[0])
        assert np.isfinite(S_est).all()

    def test_sobi_output_shape(self, sample_bss_mixture):
        """Test SOBI output dimensions."""
        S_true, X, A, fs = sample_bss_mixture
        n_obs = X.shape[0]

        # Request fewer sources than observations
        S_est, W = run_sobi(X, n_sources=2, n_lags=10)

        assert S_est.shape[0] == 2
        assert W.shape == (2, n_obs)

    def test_sobi_whitening(self, sample_bss_mixture):
        """Test that SOBI whitens the data."""
        S_true, X, A, fs = sample_bss_mixture

        S_est, W = run_sobi(X, n_sources=3, n_lags=10)

        # Check approximate decorrelation
        corr = np.corrcoef(S_est)
        off_diag = np.abs(corr[np.eye(3) == 0])
        assert np.mean(off_diag) < 0.5  # Should be somewhat decorrelated


class TestFastICA:
    """Tests for FastICA algorithm."""

    def test_basic_fastica(self, sample_bss_mixture):
        """Test basic FastICA separation."""
        S_true, X, A, fs = sample_bss_mixture
        n_sources = S_true.shape[0]

        S_est, W = run_fastica(X, n_sources=n_sources)

        assert S_est.shape[0] == n_sources
        assert S_est.shape[1] == X.shape[1]
        assert np.isfinite(S_est).all()

    def test_fastica_random_state(self, sample_bss_mixture):
        """Test FastICA reproducibility with random state."""
        S_true, X, A, fs = sample_bss_mixture

        S1, _ = run_fastica(X, n_sources=3, random_state=42)
        S2, _ = run_fastica(X, n_sources=3, random_state=42)

        np.testing.assert_array_almost_equal(S1, S2)


class TestJADE:
    """Tests for JADE algorithm."""

    def test_basic_jade(self, sample_bss_mixture):
        """Test basic JADE separation."""
        S_true, X, A, fs = sample_bss_mixture
        n_sources = S_true.shape[0]

        S_est, A_est, W = run_jade(X, n_sources=n_sources)

        assert S_est.shape[0] == n_sources
        assert S_est.shape[1] == X.shape[1]
        assert A_est.shape == (X.shape[0], n_sources)
        assert W.shape == (n_sources, X.shape[0])

    def test_jade_reconstruction(self, sample_bss_mixture):
        """Test that JADE can approximately reconstruct X."""
        S_true, X, A, fs = sample_bss_mixture

        S_est, A_est, W = run_jade(X, n_sources=3)
        X_reconstructed = A_est @ S_est

        # Reconstruction should be close to original
        relative_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
        assert relative_error < 0.5  # Allow some tolerance


class TestNMF:
    """Tests for NMF algorithm."""

    def test_basic_nmf(self):
        """Test basic NMF with non-negative data."""
        np.random.seed(42)
        X = np.abs(np.random.randn(5, 100))  # Non-negative

        S_est, A_est, W = run_nmf(X, n_sources=3)

        assert S_est.shape[0] == 3
        assert np.all(S_est >= 0)  # NMF should produce non-negative results

    def test_nmf_negative_handling(self):
        """Test NMF handles negative values by shifting."""
        np.random.seed(42)
        X = np.random.randn(5, 100)  # Contains negatives

        S_est, A_est, W = run_nmf(X, n_sources=3)

        # Should still work (shifted internally)
        assert S_est.shape[0] == 3


class TestPCA:
    """Tests for PCA algorithm."""

    def test_basic_pca(self, sample_bss_mixture):
        """Test basic PCA."""
        S_true, X, A, fs = sample_bss_mixture

        S_est, A_est, W = run_pca(X, n_sources=3)

        assert S_est.shape[0] == 3
        assert W.shape == (3, X.shape[0])

    def test_pca_whitening(self, sample_bss_mixture):
        """Test PCA with whitening."""
        S_true, X, A, fs = sample_bss_mixture

        S_est, _, _ = run_pca(X, n_sources=3, whiten=True)

        # Whitened components should have unit variance
        for i in range(S_est.shape[0]):
            assert abs(np.std(S_est[i]) - 1.0) < 0.1


class TestRunBSS:
    """Tests for run_bss unified interface."""

    def test_sobi_method(self, sample_bss_mixture):
        """Test run_bss with SOBI."""
        _, X, _, _ = sample_bss_mixture
        S_est, W = run_bss(X, method="SOBI", n_sources=3, n_lags=10)

        assert S_est.shape[0] == 3

    def test_fastica_method(self, sample_bss_mixture):
        """Test run_bss with FastICA."""
        _, X, _, _ = sample_bss_mixture
        S_est, W = run_bss(X, method="FastICA", n_sources=3)

        assert S_est.shape[0] == 3

    def test_pca_method(self, sample_bss_mixture):
        """Test run_bss with PCA."""
        _, X, _, _ = sample_bss_mixture
        S_est, W = run_bss(X, method="PCA", n_sources=3)

        assert S_est.shape[0] == 3

    def test_invalid_method(self, sample_bss_mixture):
        """Test run_bss with invalid method."""
        _, X, _, _ = sample_bss_mixture
        with pytest.raises(ValueError, match="Unknown BSS method"):
            run_bss(X, method="invalid")


class TestBSSFactory:
    """Tests for bss_factory function."""

    def test_factory_returns_tuple(self, sample_bss_mixture):
        """Test that factory returns (sources, mixing, unmixing)."""
        _, X, _, _ = sample_bss_mixture
        sources, A, W = bss_factory(X, "sobi", n_components=3, n_lags=10)

        assert sources.shape[0] == 3
        assert A.shape[1] == 3
        assert W.shape[0] == 3


class TestJointDiagonalize:
    """Tests for joint diagonalization helper."""

    def test_diagonalization(self):
        """Test joint diagonalization of known matrices."""
        np.random.seed(42)
        n = 3
        K = 5

        # Create diagonal matrices
        R_list = [np.diag(np.random.rand(n)) for _ in range(K)]

        U = _joint_diagonalize_jacobi(R_list, max_iter=100, tol=1e-6)

        # After diagonalization, U @ R @ U.T should still be diagonal
        for R in R_list:
            result = U @ R @ U.T
            # Off-diagonal should be small
            off_diag = result[np.eye(n) == 0]
            assert np.max(np.abs(off_diag)) < 0.1
