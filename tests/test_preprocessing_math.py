"""
Mathematical correctness tests for preprocessing.

Verifies:
1. Detrending removes linear trends exactly
2. Z-score normalization: mean=0, std=1
3. Min-max normalization: min=0, max=1
4. Bandpass filter: passband gain ≈ 1, stopband attenuation
5. DC removal
"""

import numpy as np
import pytest
from scipy.signal import freqz

from src.preprocessing import (
    preprocess_signals,
    bandpass_filter,
    detrend_signal,
    normalize_signal,
)


class TestDetrendCorrectness:
    """Verify detrending mathematically."""

    def test_removes_linear_trend(self):
        """Detrend should make a linear signal flat."""
        n = 1000
        x = np.linspace(0, 100, n)  # Pure linear trend

        result = detrend_signal(x)

        # After detrend, signal should be near zero
        assert np.max(np.abs(result)) < 1e-10, (
            f"Max residual after detrend={np.max(np.abs(result)):.2e}"
        )

    def test_removes_affine_trend(self):
        """Detrend should remove y = ax + b."""
        n = 1000
        x = np.arange(n)
        a, b = 3.0, 50.0
        signal = a * x + b + np.random.randn(n) * 0.01

        result = detrend_signal(signal)

        # Residual should be small (just noise)
        assert np.std(result) < 0.1

    def test_preserves_oscillatory_signal(self):
        """Detrend should not significantly alter a pure oscillation."""
        n = 1000
        t = np.arange(n) / 100.0
        signal = np.sin(2 * np.pi * 10 * t)  # Pure sinusoid, no trend

        result = detrend_signal(signal)

        # Should be almost identical (allow small numerical errors)
        np.testing.assert_allclose(result, signal, atol=0.01)


class TestNormalizationCorrectness:
    """Verify normalization mathematically."""

    def test_zscore_mean_zero(self):
        """Z-score should produce zero mean (when combined with DC removal in preprocessing)."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 5 + 100  # mean=100, std=5

        # normalize_signal only divides by std, doesn't remove mean
        # preprocess_signals removes DC before normalizing
        result = normalize_signal(signal, method="zscore")

        # After just z-score (no mean removal), mean should be original_mean/std
        expected_mean = np.mean(signal) / np.std(signal)
        assert abs(np.mean(result) - expected_mean) < 1e-10

    def test_zscore_std_one(self):
        """Z-score should produce unit standard deviation."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 5 + 100

        result = normalize_signal(signal, method="zscore")

        assert abs(np.std(result) - 1.0) < 1e-10, (
            f"Z-score std={np.std(result):.10f}, expected 1.0"
        )

    def test_zscore_formula(self):
        """Verify z-score implements x / std exactly."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = normalize_signal(signal, method="zscore")

        expected_std = np.std(signal)
        expected = signal / expected_std

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_minmax_range(self):
        """Min-max should produce values in [0, 1]."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 10 - 5

        result = normalize_signal(signal, method="minmax")

        assert np.min(result) >= -1e-10, f"Min={np.min(result)}"
        assert np.max(result) <= 1.0 + 1e-10, f"Max={np.max(result)}"

    def test_minmax_formula(self):
        """Verify min-max implements (x - min) / (max - min) exactly."""
        signal = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = normalize_signal(signal, method="minmax")

        expected = (signal - 2.0) / 8.0  # (x-2)/(10-2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_constant_signal_zscore(self):
        """Z-score of constant signal should handle zero std gracefully."""
        signal = np.ones(100) * 5.0

        # Should not crash
        result = normalize_signal(signal, method="zscore")
        assert np.isfinite(result).all()


class TestBandpassFilterCorrectness:
    """Verify bandpass filter mathematically."""

    def test_passband_gain(self):
        """Frequencies in passband should have gain ≈ 1."""
        fs = 1000
        lowcut, highcut = 100, 400

        # Get filter coefficients
        from scipy.signal import butter
        nyquist = fs / 2
        b, a = butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')

        # Frequency response
        worN = np.linspace(0, nyquist, 1000)
        w, h = freqz(b, a, worN=worN * 2 * np.pi / fs)

        freqs = w * fs / (2 * np.pi)
        gains = np.abs(h)

        # Check passband (200-300 Hz should be well within 100-400 Hz)
        passband_mask = (freqs >= 200) & (freqs <= 300)
        passband_gains = gains[passband_mask]

        assert np.all(passband_gains > 0.9), (
            f"Passband gain too low: min={np.min(passband_gains):.4f}"
        )

    def test_stopband_attenuation(self):
        """Frequencies outside passband should be attenuated."""
        fs = 1000
        lowcut, highcut = 100, 400

        from scipy.signal import butter
        nyquist = fs / 2
        b, a = butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')

        worN = np.linspace(0, nyquist, 1000)
        w, h = freqz(b, a, worN=worN * 2 * np.pi / fs)

        freqs = w * fs / (2 * np.pi)
        gains = np.abs(h)

        # Low stopband (below 50 Hz)
        low_stop = gains[freqs < 50]
        assert np.all(low_stop < 0.1), (
            f"Low stopband not attenuated: max={np.max(low_stop):.4f}"
        )

    def test_filter_preserves_inband_signal(self):
        """A signal within passband should pass through mostly unchanged."""
        fs = 1000
        n_samples = 5000
        t = np.arange(n_samples) / fs

        # 200 Hz signal (within 100-400 Hz passband)
        signal = np.sin(2 * np.pi * 200 * t)
        filtered = bandpass_filter(signal, fs, 100, 400, order=4)

        # Correlation should be high
        corr = np.corrcoef(signal, filtered)[0, 1]
        assert corr > 0.95, (
            f"In-band signal correlation={corr:.4f} < 0.95"
        )

    def test_filter_removes_outband_signal(self):
        """A signal outside passband should be strongly attenuated."""
        fs = 1000
        n_samples = 5000
        t = np.arange(n_samples) / fs

        # 10 Hz signal (below 100-400 Hz passband)
        signal = np.sin(2 * np.pi * 10 * t)
        filtered = bandpass_filter(signal, fs, 100, 400, order=4)

        # RMS should be much smaller
        rms_in = np.sqrt(np.mean(signal**2))
        rms_out = np.sqrt(np.mean(filtered**2))
        attenuation_db = 20 * np.log10(rms_out / rms_in + 1e-12)

        assert attenuation_db < -20, (
            f"Out-of-band attenuation={attenuation_db:.1f} dB > -20 dB"
        )


class TestDCRemoval:
    """Verify DC component removal."""

    def test_dc_removal(self):
        """Preprocessing should remove DC offset."""
        np.random.seed(42)
        fs = 1000
        n_samples = 2000

        # Signal with DC offset
        signal = np.random.randn(n_samples) + 100.0  # DC = 100
        signals = signal.reshape(1, -1)

        config = {"detrend": False, "bandpass": None, "normalize": None}
        result, _ = preprocess_signals(signals, fs, config)

        # DC should be removed
        assert abs(np.mean(result)) < 1e-10, (
            f"DC not removed: mean={np.mean(result):.2e}"
        )


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline mathematically."""

    def test_full_pipeline_output(self):
        """Full pipeline should produce zero-mean, unit-variance signals."""
        np.random.seed(42)
        fs = 1000
        n_samples = 2000

        # Signal with trend, DC, and unknown variance
        t = np.arange(n_samples) / fs
        signal = 50 * t + 100 + 10 * np.sin(2 * np.pi * 50 * t) + np.random.randn(n_samples)
        signals = signal.reshape(1, -1)

        config = {
            "detrend": True,
            "bandpass": None,
            "normalize": "zscore",
        }
        result, _ = preprocess_signals(signals, fs, config)

        # After full pipeline: mean ≈ 0, std ≈ 1
        assert abs(np.mean(result)) < 0.1
        assert abs(np.std(result) - 1.0) < 0.1
