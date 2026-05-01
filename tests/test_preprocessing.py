"""
Tests for preprocessing module.
"""

import numpy as np
import pytest
from src.preprocessing import preprocess_signals, bandpass_filter


class TestPreprocessSignals:
    """Tests for preprocess_signals function."""

    def test_basic_preprocessing(self, sample_signals_multi):
        """Test basic preprocessing pipeline."""
        signals, fs = sample_signals_multi
        config = {
            "detrend": True,
            "bandpass": None,
            "normalize": "zscore",
        }

        result, fs_out = preprocess_signals(signals, fs, config)

        assert result.shape == signals.shape
        assert fs_out == fs

    def test_detrend(self, sample_signals_multi):
        """Test detrending removes linear trend."""
        signals, fs = sample_signals_multi
        # Add a linear trend
        trend = np.linspace(0, 10, signals.shape[1])
        signals_with_trend = signals + trend

        config = {"detrend": True, "bandpass": None, "normalize": None}
        result, _ = preprocess_signals(signals_with_trend, fs, config)

        # Check that trend is removed (mean should be close to 0)
        for i in range(result.shape[0]):
            assert abs(np.mean(result[i])) < 1e-10

    def test_zscore_normalization(self, sample_signals_multi):
        """Test z-score normalization."""
        signals, fs = sample_signals_multi
        config = {"detrend": False, "bandpass": None, "normalize": "zscore"}

        result, _ = preprocess_signals(signals, fs, config)

        # After z-score, each channel should have std ≈ 1
        for i in range(result.shape[0]):
            assert abs(np.std(result[i]) - 1.0) < 1e-10

    def test_minmax_normalization(self, sample_signals_multi):
        """Test min-max normalization."""
        signals, fs = sample_signals_multi
        config = {"detrend": False, "bandpass": None, "normalize": "minmax"}

        result, _ = preprocess_signals(signals, fs, config)

        # After min-max, values should be in [0, 1]
        for i in range(result.shape[0]):
            assert np.min(result[i]) >= -1e-10
            assert np.max(result[i]) <= 1.0 + 1e-10

    def test_bandpass_filter(self, sample_signals_multi):
        """Test bandpass filtering."""
        signals, fs = sample_signals_multi
        config = {"detrend": False, "bandpass": (50, 200), "normalize": None}

        result, _ = preprocess_signals(signals, fs, config)

        # Shape should be preserved
        assert result.shape == signals.shape

    def test_resampling(self, sample_signals_multi):
        """Test resampling to lower frequency."""
        signals, fs = sample_signals_multi
        new_fs = 500
        config = {"detrend": False, "bandpass": None, "normalize": None, "resample_fs": new_fs}

        result, fs_out = preprocess_signals(signals, fs, config)

        assert fs_out == new_fs
        # Check approximate length
        expected_samples = int(signals.shape[1] * new_fs / fs)
        assert abs(result.shape[1] - expected_samples) <= 1

    def test_none_config(self, sample_signals_multi):
        """Test with None config uses defaults."""
        signals, fs = sample_signals_multi
        result, fs_out = preprocess_signals(signals, fs, None)

        assert result.shape == signals.shape
        assert fs_out == fs

    def test_empty_config(self, sample_signals_multi):
        """Test with empty config."""
        signals, fs = sample_signals_multi
        result, fs_out = preprocess_signals(signals, fs, {})

        assert result.shape == signals.shape
        assert fs_out == fs


class TestBandpassFilter:
    """Tests for bandpass_filter function."""

    def test_basic_filter(self, sample_signal_1d):
        """Test basic bandpass filtering."""
        fs = 1000
        result = bandpass_filter(sample_signal_1d, fs, 100, 400)

        assert result.shape == sample_signal_1d.shape
        assert np.isfinite(result).all()

    def test_filter_preserves_length(self, sample_signal_1d):
        """Test that filter preserves signal length."""
        fs = 1000
        result = bandpass_filter(sample_signal_1d, fs, 100, 400)

        assert len(result) == len(sample_signal_1d)

    def test_filter_order(self, sample_signal_1d):
        """Test different filter orders."""
        fs = 1000
        for order in [2, 4, 6]:
            result = bandpass_filter(sample_signal_1d, fs, 100, 400, order=order)
            assert result.shape == sample_signal_1d.shape
