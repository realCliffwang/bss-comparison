"""
Tests for evaluation module.
"""

import numpy as np
import pytest
from src.evaluation import (
    compute_metrics,
    compute_independence_metric,
    compute_fault_detection_score,
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_separation(self):
        """Test metrics with perfect separation."""
        np.random.seed(42)
        S_true = np.random.randn(3, 1000)
        S_est = S_true.copy()

        metrics = compute_metrics(S_true, S_est)

        assert "SIR_dB" in metrics
        assert "mean_correlation" in metrics
        assert metrics["mean_correlation"] > 0.99

    def test_noisy_separation(self):
        """Test metrics with noisy separation."""
        np.random.seed(42)
        S_true = np.random.randn(3, 1000)
        S_est = S_true + 0.1 * np.random.randn(3, 1000)

        metrics = compute_metrics(S_true, S_est)

        assert metrics["mean_correlation"] > 0.8

    def test_different_sizes(self):
        """Test metrics with different source counts."""
        np.random.seed(42)
        S_true = np.random.randn(3, 1000)
        S_est = np.random.randn(2, 1000)

        metrics = compute_metrics(S_true, S_est)

        assert "SIR_dB" in metrics


class TestComputeIndependenceMetric:
    """Tests for compute_independence_metric function."""

    def test_independent_sources(self):
        """Test independence metric with independent sources."""
        np.random.seed(42)
        S = np.random.randn(3, 1000)

        metric = compute_independence_metric(S)

        # Should be low (close to 0) for independent sources
        assert metric < 0.2

    def test_correlated_sources(self):
        """Test independence metric with correlated sources."""
        np.random.seed(42)
        S = np.random.randn(2, 1000)
        S[1] = S[0] + 0.5 * np.random.randn(1000)  # Correlated

        metric = compute_independence_metric(S)

        # Should be higher for correlated sources
        assert metric > 0.3

    def test_single_source(self):
        """Test independence metric with single source."""
        S = np.random.randn(1, 1000)

        metric = compute_independence_metric(S)

        assert metric == 0.0


class TestComputeFaultDetectionScore:
    """Tests for compute_fault_detection_score function."""

    def test_basic_ffds(self):
        """Test basic FFDS computation."""
        np.random.seed(42)
        fs = 1000
        S = np.random.randn(3, 5000)
        fault_freqs = {"BPFO": 100, "BPFI": 150}

        score = compute_fault_detection_score(S, fs, fault_freqs)

        assert isinstance(score, float)
        assert score >= 0

    def test_no_fault_freqs(self):
        """Test FFDS with no fault frequencies."""
        S = np.random.randn(3, 5000)
        fs = 1000

        score = compute_fault_detection_score(S, fs, {})

        assert score == 0.0
