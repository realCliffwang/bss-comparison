"""
Tests for feature extraction module.
"""

import numpy as np
import pytest
from src.feature_extractor import (
    extract_time_domain_features,
    extract_freq_domain_features,
    extract_wpt_features,
    extract_stft_features,
    extract_features,
)


class TestTimeDomainFeatures:
    """Tests for time domain feature extraction."""

    def test_basic_extraction(self, sample_signal_1d):
        """Test basic time domain feature extraction."""
        features = extract_time_domain_features(sample_signal_1d)

        assert features.shape == (10,)
        assert np.isfinite(features).all()

    def test_feature_names(self, sample_signal_1d):
        """Test that all expected features are extracted."""
        features = extract_time_domain_features(sample_signal_1d)

        # Features: mean, var, rms, peak, peak_to_peak, kurtosis, skewness,
        # waveform_factor, impulse_factor, clearance_factor
        assert len(features) == 10

    def test_rms_positive(self, sample_signal_1d):
        """Test that RMS is always positive."""
        features = extract_time_domain_features(sample_signal_1d)
        rms = features[2]

        assert rms >= 0

    def test_peak_positive(self, sample_signal_1d):
        """Test that peak value is always positive."""
        features = extract_time_domain_features(sample_signal_1d)
        peak = features[3]

        assert peak >= 0

    def test_kurtosis_gaussian(self):
        """Test kurtosis of Gaussian signal is close to 3."""
        np.random.seed(42)
        signal = np.random.randn(10000)

        features = extract_time_domain_features(signal)
        kurtosis = features[5]

        # Excess kurtosis of Gaussian is 0 (scipy returns excess)
        assert abs(kurtosis) < 0.5

    def test_waveform_factor_range(self, sample_signal_1d):
        """Test waveform factor is in valid range."""
        features = extract_time_domain_features(sample_signal_1d)
        waveform_factor = features[7]

        # Waveform factor should be >= 1
        assert waveform_factor >= 1.0


class TestFreqDomainFeatures:
    """Tests for frequency domain feature extraction."""

    def test_basic_extraction(self, sample_signal_1d):
        """Test basic frequency domain feature extraction."""
        fs = 1000
        features = extract_freq_domain_features(sample_signal_1d, fs)

        assert features.shape == (5,)
        assert np.isfinite(features).all()

    def test_spectral_centroid_positive(self, sample_signal_1d):
        """Test that spectral centroid is positive."""
        fs = 1000
        features = extract_freq_domain_features(sample_signal_1d, fs)
        centroid = features[0]

        assert centroid > 0

    def test_dominant_freq_ratio(self, sample_signal_1d):
        """Test dominant frequency energy ratio is between 0 and 1."""
        fs = 1000
        features = extract_freq_domain_features(sample_signal_1d, fs)
        dom_ratio = features[2]

        assert 0 <= dom_ratio <= 1

    def test_known_frequency(self):
        """Test feature extraction on signal with known frequency."""
        fs = 1000
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 100 * t)  # 100 Hz

        features = extract_freq_domain_features(signal, fs)
        centroid = features[0]

        # Centroid should be near 100 Hz
        assert abs(centroid - 100) < 50


class TestWPTFeatures:
    """Tests for WPT feature extraction."""

    def test_basic_extraction(self, sample_signal_1d):
        """Test basic WPT feature extraction."""
        fs = 1000
        features = extract_wpt_features(sample_signal_1d, fs)

        # 2^max_level features (default max_level=4)
        assert features.shape == (16,)
        assert np.isfinite(features).all()

    def test_normalized_energies(self, sample_signal_1d):
        """Test that WPT features are normalized."""
        fs = 1000
        features = extract_wpt_features(sample_signal_1d, fs)

        # Should sum to approximately 1
        assert abs(np.sum(features) - 1.0) < 0.01

    def test_different_levels(self, sample_signal_1d):
        """Test WPT with different decomposition levels."""
        fs = 1000
        for level in [2, 3, 4]:
            features = extract_wpt_features(sample_signal_1d, fs, max_level=level)
            assert features.shape == (2**level,)


class TestSTFTFeatures:
    """Tests for STFT feature extraction."""

    def test_basic_extraction(self, sample_signal_1d):
        """Test basic STFT feature extraction."""
        fs = 1000
        features = extract_stft_features(sample_signal_1d, fs)

        assert features.shape == (32,)  # Default n_freq_bands
        assert np.isfinite(features).all()

    def test_normalized_energies(self, sample_signal_1d):
        """Test that STFT features are normalized."""
        fs = 1000
        features = extract_stft_features(sample_signal_1d, fs)

        # Should sum to approximately 1
        assert abs(np.sum(features) - 1.0) < 0.01


class TestExtractFeatures:
    """Tests for unified extract_features function."""

    def test_all_features(self, sample_signal_1d):
        """Test extraction of all features."""
        fs = 1000
        features = extract_features(sample_signal_1d, fs, feature_set="all")

        # time(10) + freq(5) + wpt(16) + stft(32) = 63
        assert features.shape == (63,)
        assert np.isfinite(features).all()

    def test_time_only(self, sample_signal_1d):
        """Test time features only."""
        fs = 1000
        features = extract_features(sample_signal_1d, fs, feature_set="time")

        assert features.shape == (10,)

    def test_freq_only(self, sample_signal_1d):
        """Test frequency features only."""
        fs = 1000
        features = extract_features(sample_signal_1d, fs, feature_set="freq")

        assert features.shape == (5,)

    def test_time_freq_only(self, sample_signal_1d):
        """Test time-frequency features only."""
        fs = 1000
        features = extract_features(sample_signal_1d, fs, feature_set="time_freq")

        assert features.shape == (48,)  # wpt(16) + stft(32)

    def test_invalid_feature_set(self, sample_signal_1d):
        """Test invalid feature set raises error."""
        fs = 1000
        with pytest.raises(ValueError, match="Unknown feature_set"):
            extract_features(sample_signal_1d, fs, feature_set="invalid")
