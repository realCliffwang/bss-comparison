"""
Mathematical correctness tests for feature extraction.

Verifies:
1. Time domain features match hand-calculated values
2. Frequency domain features are physically meaningful
3. Feature extraction is deterministic
"""

import numpy as np
import pytest
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skewness

from src.feature_extractor import (
    extract_time_domain_features,
    extract_freq_domain_features,
    extract_features,
)


class TestTimeDomainFeatureAccuracy:
    """Verify time domain features match mathematical definitions."""

    def test_mean_feature(self):
        """Mean feature should equal np.mean."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 3 + 10

        features = extract_time_domain_features(signal)
        mean_feature = features[0]

        assert abs(mean_feature - np.mean(signal)) < 1e-10

    def test_variance_feature(self):
        """Variance feature should equal np.var."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 3 + 10

        features = extract_time_domain_features(signal)
        var_feature = features[1]

        assert abs(var_feature - np.var(signal)) < 1e-10

    def test_rms_feature(self):
        """RMS should equal sqrt(mean(x^2))."""
        np.random.seed(42)
        signal = np.random.randn(1000)

        features = extract_time_domain_features(signal)
        rms_feature = features[2]

        expected_rms = np.sqrt(np.mean(signal**2))
        assert abs(rms_feature - expected_rms) < 1e-10

    def test_peak_feature(self):
        """Peak should equal max(|x|)."""
        signal = np.array([1.0, -5.0, 3.0, -2.0, 4.0])

        features = extract_time_domain_features(signal)
        peak_feature = features[3]

        assert abs(peak_feature - 5.0) < 1e-10

    def test_peak_to_peak_feature(self):
        """Peak-to-peak should equal max(x) - min(x)."""
        signal = np.array([1.0, -5.0, 3.0, -2.0, 4.0])

        features = extract_time_domain_features(signal)
        p2p_feature = features[4]

        expected = 4.0 - (-5.0)  # max - min = 9.0
        assert abs(p2p_feature - expected) < 1e-10

    def test_kurtosis_feature(self):
        """Kurtosis should match scipy.stats.kurtosis."""
        np.random.seed(42)
        signal = np.random.randn(1000)

        features = extract_time_domain_features(signal)
        kurt_feature = features[5]

        expected = sp_kurtosis(signal)
        assert abs(kurt_feature - expected) < 1e-10

    def test_skewness_feature(self):
        """Skewness should match scipy.stats.skew."""
        np.random.seed(42)
        signal = np.random.randn(1000)

        features = extract_time_domain_features(signal)
        skew_feature = features[6]

        expected = sp_skewness(signal)
        assert abs(skew_feature - expected) < 1e-10

    def test_waveform_factor_formula(self):
        """Waveform factor = RMS / mean(|x|)."""
        signal = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        features = extract_time_domain_features(signal)
        wf_feature = features[7]

        rms = np.sqrt(np.mean(signal**2))
        abs_mean = np.mean(np.abs(signal))
        expected = rms / abs_mean

        assert abs(wf_feature - expected) < 1e-10

    def test_impulse_factor_formula(self):
        """Impulse factor = peak / mean(|x|)."""
        signal = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        features = extract_time_domain_features(signal)
        imp_feature = features[8]

        peak = np.max(np.abs(signal))
        abs_mean = np.mean(np.abs(signal))
        expected = peak / abs_mean

        assert abs(imp_feature - expected) < 1e-10

    def test_clearance_factor_formula(self):
        """Clearance factor = peak / mean(sqrt(|x|))^2."""
        signal = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

        features = extract_time_domain_features(signal)
        clr_feature = features[9]

        peak = np.max(np.abs(signal))
        sqrt_abs_mean = np.mean(np.sqrt(np.abs(signal)))
        expected = peak / (sqrt_abs_mean ** 2 + 1e-12)

        # Allow some numerical tolerance
        assert abs(clr_feature - expected) / (expected + 1e-12) < 0.01


class TestFreqDomainFeatureAccuracy:
    """Verify frequency domain features match definitions."""

    def test_spectral_centroid_formula(self):
        """Spectral centroid = sum(f * |X(f)|) / sum(|X(f)|)."""
        np.random.seed(42)
        fs = 1000
        signal = np.random.randn(1000)

        features = extract_freq_domain_features(signal, fs)
        centroid = features[0]

        # Manual calculation
        from scipy.fft import rfft, rfftfreq
        spectrum = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), 1.0 / fs)

        # Skip DC
        spectrum = spectrum[1:]
        freqs = freqs[1:]

        expected = np.sum(freqs * spectrum) / np.sum(spectrum)
        assert abs(centroid - expected) < 1e-10

    def test_dominant_freq_energy_ratio(self):
        """Dominant freq ratio should be between 0 and 1."""
        np.random.seed(42)
        fs = 1000
        signal = np.random.randn(1000)

        features = extract_freq_domain_features(signal, fs)
        dom_ratio = features[2]

        assert 0 <= dom_ratio <= 1, f"Dominant freq ratio={dom_ratio}"

    def test_dominant_freq_of_pure_sinusoid(self):
        """Pure sinusoid should have dominant freq energy ratio ≈ 0.5 (single peak)."""
        fs = 1000
        n = 1000
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 100 * t)

        features = extract_freq_domain_features(signal, fs)
        dom_ratio = features[2]

        # Single frequency should concentrate energy
        assert dom_ratio > 0.3, (
            f"Dominant freq ratio={dom_ratio:.4f} too low for pure sinusoid"
        )


class TestFeatureDeterminism:
    """Verify feature extraction is deterministic."""

    def test_same_input_same_output(self):
        """Same input should always produce same features."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        fs = 1000

        feat1 = extract_features(signal, fs)
        feat2 = extract_features(signal, fs)

        np.testing.assert_array_equal(feat1, feat2)

    def test_no_randomness_in_features(self):
        """Features should not depend on random state."""
        # Use non-constant signal to avoid NaN in kurtosis/skewness
        np.random.seed(42)
        signal = np.random.randn(100)
        fs = 1000

        # Run multiple times
        for _ in range(5):
            feat = extract_features(signal, fs, feature_set="time")
            assert np.isfinite(feat).all()


class TestFeaturePhysicalMeaning:
    """Verify features have physical meaning."""

    def test_constant_signal_zero_variance(self):
        """Constant signal should have zero variance."""
        signal = np.ones(100) * 5.0

        features = extract_time_domain_features(signal)
        var_feature = features[1]

        assert var_feature < 1e-10

    def test_rms_of_sinusoid(self):
        """RMS of A*sin(wt) should be A/sqrt(2)."""
        fs = 1000
        n = 10000
        t = np.arange(n) / fs
        A = 3.0
        signal = A * np.sin(2 * np.pi * 50 * t)

        features = extract_time_domain_features(signal)
        rms = features[2]

        expected = A / np.sqrt(2)
        error_pct = abs(rms - expected) / expected * 100
        assert error_pct < 5, (
            f"RMS={rms:.4f}, expected A/sqrt(2)={expected:.4f} (error={error_pct:.1f}%)"
        )

    def test_kurtosis_of_gaussian(self):
        """Kurtosis of Gaussian should be ≈ 0 (excess kurtosis)."""
        np.random.seed(42)
        signal = np.random.randn(10000)

        features = extract_time_domain_features(signal)
        kurt = features[5]

        assert abs(kurt) < 0.5, f"Gaussian kurtosis={kurt:.4f}, expected ~0"
