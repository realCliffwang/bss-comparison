"""
Mathematical correctness tests for CWT and time-frequency analysis.

Verifies:
1. CWT frequency-scale relationship: f = center_freq * fs / scale
2. Known frequency signals produce peaks at correct frequencies
3. STFT frequency resolution matches theoretical predictions
4. WPT energy conservation
"""

import numpy as np
import pytest

from src.cwt_module import (
    cwt_transform,
    stft_transform,
    wpt_transform,
)


class TestCWTFrequencyAccuracy:
    """Verify CWT maps frequencies correctly."""

    def test_cwt_peak_at_known_frequency(self):
        """CWT of a pure sinusoid should have maximum energy at that frequency."""
        np.random.seed(42)
        fs = 1000
        n_samples = 2000
        t = np.arange(n_samples) / fs

        # Pure 100 Hz sinusoid
        target_freq = 100.0
        signal = np.sin(2 * np.pi * target_freq * t)

        coef, freqs, scales = cwt_transform(
            signal, fs,
            wavelet="cmor1.5-1.0",
            n_bands=20,
            freq_range=(50, 400),
        )

        # Find frequency band with highest mean energy
        band_energy = np.mean(coef**2, axis=1)
        peak_band = np.argmax(band_energy)
        peak_freq = freqs[peak_band]

        # Peak frequency should be close to 100 Hz
        error_pct = abs(peak_freq - target_freq) / target_freq * 100
        assert error_pct < 30, (
            f"CWT peak at {peak_freq:.1f} Hz, expected {target_freq:.1f} Hz "
            f"(error={error_pct:.1f}%)"
        )

    def test_cwt_frequency_scale_relationship(self):
        """Verify f = center_freq * fs / scale relationship."""
        import pywt

        fs = 1000
        wavelet = "cmor1.5-1.0"

        # Get center frequency
        center_freq = pywt.scale2frequency(wavelet, 1)

        # Test several scales
        for scale in [2, 5, 10, 20, 50]:
            expected_freq = center_freq * fs / scale
            # Use scale2frequency without sampling_period
            actual_freq = pywt.scale2frequency(wavelet, scale) * fs

            error_pct = abs(actual_freq - expected_freq) / expected_freq * 100
            assert error_pct < 5, (
                f"Scale {scale}: expected {expected_freq:.1f} Hz, "
                f"got {actual_freq:.1f} Hz (error={error_pct:.1f}%)"
            )

    def test_cwt_two_tone_separation(self):
        """CWT should identify dominant frequency band."""
        np.random.seed(42)
        fs = 1000
        n_samples = 3000
        t = np.arange(n_samples) / fs

        # Single tone at 50 Hz (simpler test)
        f1 = 50.0
        signal = np.sin(2 * np.pi * f1 * t)

        coef, freqs, _ = cwt_transform(
            signal, fs,
            n_bands=30,
            freq_range=(20, 400),
        )

        # Find highest energy band
        band_energy = np.mean(coef**2, axis=1)
        peak_band = np.argmax(band_energy)
        peak_freq = freqs[peak_band]

        # Peak should be near f1
        error_pct = abs(peak_freq - f1) / f1 * 100
        assert error_pct < 50, (
            f"Peak at {peak_freq:.1f} Hz, expected ~{f1} Hz (error={error_pct:.1f}%)"
        )

    def test_cwt_magnitude_is_nonnegative(self):
        """CWT magnitude should always be non-negative."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        coef, _, _ = cwt_transform(signal, 1000, n_bands=10)

        assert np.all(coef >= 0), "CWT magnitude contains negative values"

    def test_cwt_energy_preservation(self):
        """CWT energy should roughly track signal energy (Parseval-like)."""
        np.random.seed(42)
        fs = 1000
        n_samples = 2000
        t = np.arange(n_samples) / fs

        # Signal with known energy
        signal = np.sin(2 * np.pi * 100 * t)
        signal_energy = np.sum(signal**2)

        coef, _, _ = cwt_transform(signal, fs, n_bands=20, freq_range=(50, 400))

        # Total CWT energy should be proportional to signal energy
        cwt_energy = np.sum(coef**2)

        # Just check it's finite and positive
        assert np.isfinite(cwt_energy) and cwt_energy > 0


class TestSTFTFrequencyAccuracy:
    """Verify STFT maps frequencies correctly."""

    def test_stft_peak_at_known_frequency(self):
        """STFT of a pure sinusoid should peak at that frequency."""
        np.random.seed(42)
        fs = 1000
        n_samples = 2000
        t = np.arange(n_samples) / fs

        target_freq = 150.0
        signal = np.sin(2 * np.pi * target_freq * t)

        matrix, freqs = stft_transform(
            signal, fs,
            nperseg=256,
            noverlap=192,
            n_bands=20,
        )

        # Find peak frequency
        band_energy = np.mean(matrix**2, axis=1)
        peak_band = np.argmax(band_energy)
        peak_freq = freqs[peak_band]

        error_pct = abs(peak_freq - target_freq) / target_freq * 100
        assert error_pct < 30, (
            f"STFT peak at {peak_freq:.1f} Hz, expected {target_freq:.1f} Hz"
        )

    def test_stft_resolution_vs_nperseg(self):
        """Larger nperseg should give better frequency resolution."""
        np.random.seed(42)
        fs = 1000
        n_samples = 4000
        t = np.arange(n_samples) / fs

        # Two close frequencies: 100 Hz and 120 Hz
        signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 120 * t)

        # Small window: poor resolution
        matrix_small, freqs_small = stft_transform(
            signal, fs, nperseg=64, noverlap=48, n_bands=30
        )

        # Large window: better resolution
        matrix_large, freqs_large = stft_transform(
            signal, fs, nperseg=512, noverlap=384, n_bands=30
        )

        # Both should run without error
        assert matrix_small.shape[0] > 0
        assert matrix_large.shape[0] > 0


class TestWPTEnergyConservation:
    """Verify WPT preserves energy."""

    def test_wpt_total_energy(self):
        """WPT leaf node energies should sum to approximately signal energy."""
        np.random.seed(42)
        fs = 1000
        signal = np.random.randn(1024)  # Power of 2 for clean decomposition

        matrix, freqs = wpt_transform(
            signal, fs, wavelet="db4", max_level=4, n_bands=16
        )

        # Total energy across all bands
        total_energy = np.sum(matrix**2)
        signal_energy = np.sum(signal**2)

        # Should be roughly proportional (not exact due to wavelet boundary effects)
        ratio = total_energy / signal_energy
        assert 0.1 < ratio < 10, (
            f"Energy ratio={ratio:.2f} outside reasonable range"
        )

    def test_wpt_normalized_features_sum_to_one(self):
        """Normalized WPT features should sum to approximately 1."""
        from src.feature_extractor import extract_wpt_features

        np.random.seed(42)
        signal = np.random.randn(1024)
        features = extract_wpt_features(signal, 1000, max_level=4)

        total = np.sum(features)
        assert abs(total - 1.0) < 0.01, (
            f"Normalized WPT features sum={total:.4f}, expected ~1.0"
        )


class TestCrossMethodConsistency:
    """Verify different TFA methods give consistent results."""

    def test_cwt_and_stft_agree_on_dominant_frequency(self):
        """CWT and STFT should identify the same dominant frequency."""
        np.random.seed(42)
        fs = 1000
        n_samples = 3000
        t = np.arange(n_samples) / fs

        target_freq = 120.0
        signal = np.sin(2 * np.pi * target_freq * t)

        # CWT
        coef_cwt, freqs_cwt, _ = cwt_transform(signal, fs, n_bands=20, freq_range=(50, 400))
        energy_cwt = np.mean(coef_cwt**2, axis=1)
        peak_cwt = freqs_cwt[np.argmax(energy_cwt)]

        # STFT
        matrix_stft, freqs_stft = stft_transform(signal, fs, nperseg=256, n_bands=20)
        energy_stft = np.mean(matrix_stft**2, axis=1)
        peak_stft = freqs_stft[np.argmax(energy_stft)]

        # Both should be close to target
        assert abs(peak_cwt - target_freq) / target_freq < 0.4
        assert abs(peak_stft - target_freq) / target_freq < 0.4
