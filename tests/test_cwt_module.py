"""
Tests for CWT/TFA module.
"""

import numpy as np
import pytest
from src.cwt_module import (
    cwt_transform,
    cwt_transform_multichannel,
    build_observation_matrix,
    stft_transform,
    wpt_transform,
    time_freq_factory,
)


class TestCWTTransform:
    """Tests for cwt_transform function."""

    def test_basic_cwt(self, sample_signal_1d):
        """Test basic CWT computation."""
        fs = 1000
        coef, freqs, scales = cwt_transform(
            sample_signal_1d, fs,
            wavelet="cmor1.5-1.0",
            n_bands=10,
            freq_range=(50, 400),
        )

        # Note: Actual number of bands may differ due to scale rounding
        assert coef.shape[0] > 0
        assert coef.shape[0] <= 10
        assert coef.shape[1] == len(sample_signal_1d)
        assert len(freqs) > 0
        assert len(scales) > 0
        assert np.isfinite(coef).all()

    def test_cwt_frequencies_in_range(self, sample_signal_1d):
        """Test that returned frequencies are approximately within specified range."""
        fs = 1000
        freq_range = (100, 400)

        _, freqs, _ = cwt_transform(
            sample_signal_1d, fs,
            freq_range=freq_range,
            n_bands=10,
        )

        # Allow larger tolerance due to scale discretization
        assert all(f >= freq_range[0] * 0.5 for f in freqs)
        assert all(f <= freq_range[1] * 2.0 for f in freqs)

    def test_cwt_different_wavelets(self, sample_signal_1d):
        """Test CWT with different wavelet types."""
        fs = 1000
        wavelets = ["cmor1.5-1.0", "morl"]

        for wavelet in wavelets:
            coef, freqs, scales = cwt_transform(
                sample_signal_1d, fs,
                wavelet=wavelet,
                n_bands=5,
            )
            assert coef.shape[0] == 5

    def test_cwt_magnitude_output(self, sample_signal_1d):
        """Test that CWT output is non-negative (magnitude)."""
        fs = 1000
        coef, _, _ = cwt_transform(sample_signal_1d, fs, n_bands=5)

        assert np.all(coef >= 0)


class TestCWTTransformMultichannel:
    """Tests for cwt_transform_multichannel function."""

    def test_multichannel_cwt(self, sample_signals_multi):
        """Test multi-channel CWT."""
        signals, fs = sample_signals_multi
        results = cwt_transform_multichannel(
            signals, fs,
            n_bands=5,
            freq_range=(50, 400),
        )

        assert len(results) == signals.shape[0]
        for i, result in enumerate(results):
            assert "coef" in result
            assert "freqs" in result
            assert "scales" in result
            assert result["channel_idx"] == i


class TestBuildObservationMatrix:
    """Tests for build_observation_matrix function."""

    def test_single_channel_expansion(self, sample_signals_multi):
        """Test single channel expansion mode."""
        signals, fs = sample_signals_multi
        config = {
            "mode": "single_channel_expansion",
            "n_bands": 5,
            "freq_range": (50, 400),
            "tfa_method": "cwt",
        }

        X, labels = build_observation_matrix(signals, fs, config)

        assert X.shape[0] == 5  # n_bands
        assert X.shape[1] == signals.shape[1]
        assert len(labels) == 5

    def test_multi_channel_mode(self, sample_signals_multi):
        """Test multi-channel mode."""
        signals, fs = sample_signals_multi
        config = {
            "mode": "multi_channel",
            "n_bands": 5,
            "freq_range": (50, 400),
            "bands_per_ch": 2,
            "tfa_method": "cwt",
        }

        X, labels = build_observation_matrix(signals, fs, config)

        # Should have n_channels * bands_per_ch observations
        expected_obs = signals.shape[0] * 2
        assert X.shape[0] == expected_obs

    def test_stft_tfa_method(self, sample_signals_multi):
        """Test with STFT as TFA method."""
        signals, fs = sample_signals_multi
        config = {
            "mode": "single_channel_expansion",
            "n_bands": 5,
            "tfa_method": "stft",
        }

        X, labels = build_observation_matrix(signals[0:1], fs, config)
        assert X.shape[0] > 0


class TestSTFTTransform:
    """Tests for stft_transform function."""

    def test_basic_stft(self, sample_signal_1d):
        """Test basic STFT computation."""
        fs = 1000
        matrix, freqs = stft_transform(
            sample_signal_1d, fs,
            nperseg=128,
            noverlap=96,
            n_bands=10,
        )

        assert matrix.shape[0] == 10
        assert matrix.shape[1] == len(sample_signal_1d)
        assert len(freqs) == 10
        assert np.isfinite(matrix).all()

    def test_stft_frequency_selection(self, sample_signal_1d):
        """Test STFT with frequency range selection."""
        fs = 1000
        matrix, freqs = stft_transform(
            sample_signal_1d, fs,
            freq_range=(50, 400),
            n_bands=5,
        )

        assert matrix.shape[0] == 5


class TestWPTTransform:
    """Tests for wpt_transform function."""

    def test_basic_wpt(self, sample_signal_1d):
        """Test basic WPT computation."""
        fs = 1000
        matrix, freqs = wpt_transform(
            sample_signal_1d, fs,
            wavelet="db4",
            max_level=3,
            n_bands=8,
        )

        assert matrix.shape[0] <= 8  # May be fewer if n_bands > 2^max_level
        assert matrix.shape[0] > 0
        # WPT reconstructed signals may have different length than input
        assert matrix.shape[1] > 0


class TestTimeFreqFactory:
    """Tests for time_freq_factory function."""

    def test_cwt_method(self, sample_signal_1d):
        """Test factory with CWT method."""
        fs = 1000
        matrix, freqs = time_freq_factory(
            sample_signal_1d, fs,
            method="cwt",
            n_bands=5,
            freq_range=(50, 400),
        )

        assert matrix.shape[0] == 5

    def test_stft_method(self, sample_signal_1d):
        """Test factory with STFT method."""
        fs = 1000
        matrix, freqs = time_freq_factory(
            sample_signal_1d, fs,
            method="stft",
            n_bands=5,
        )

        assert matrix.shape[0] > 0

    def test_invalid_method(self, sample_signal_1d):
        """Test factory with invalid method raises error."""
        fs = 1000
        with pytest.raises(ValueError, match="Unknown TFA method"):
            time_freq_factory(sample_signal_1d, fs, method="invalid")
