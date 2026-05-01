"""
Feature extraction for vibration-based fault diagnosis.

Reference implementations:
  - Time/freq domain features + SVM/KNN/XGBoost on CWRU:
    https://github.com/LGDiMaggio/CWRU-bearing-fault-classification-ML
  - WPT + FFT hybrid features:
    https://github.com/Western-OC2-Lab/Vibration-Based-Fault-Diagnosis-with-Low-Delay
    (IEEE Trans. Instrumentation & Measurement, 2022)

Feature sets:
  - Time domain (10-dim): mean, variance, RMS, peak, peak-to-peak,
    kurtosis, skewness, waveform factor, impulse factor, clearance factor
  - Frequency domain (5-dim): spectral centroid, spectral variance,
    dominant freq energy ratio, spectral kurtosis, top-3 band energy ratio
  - Time-frequency domain (variable-dim): WPT/STFT node normalized energies
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skewness


def extract_time_domain_features(signal):
    """
    Extract 10 time-domain statistical features.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D raw signal.

    Returns
    -------
    features : ndarray (10,)
        [mean, var, rms, peak, peak_to_peak, kurtosis, skewness,
         waveform_factor, impulse_factor, clearance_factor]
    """
    sig = signal.astype(np.float64)
    n = len(sig)

    mean_val = np.mean(sig)
    var_val = np.var(sig)
    std_val = np.sqrt(var_val + 1e-12)
    rms_val = np.sqrt(np.mean(sig**2))
    peak_val = np.max(np.abs(sig))
    peak_to_peak = np.max(sig) - np.min(sig)
    kurt_val = sp_kurtosis(sig)
    skew_val = sp_skewness(sig)

    abs_mean = np.mean(np.abs(sig))
    waveform_factor = rms_val / (abs_mean + 1e-12)
    impulse_factor = peak_val / (abs_mean + 1e-12)
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(sig)))
    clearance_factor = peak_val / (sqrt_abs_mean ** 2 + 1e-12)

    return np.array([
        mean_val, var_val, rms_val, peak_val, peak_to_peak,
        kurt_val, skew_val,
        waveform_factor, impulse_factor, clearance_factor
    ])


def extract_freq_domain_features(signal, fs):
    """
    Extract 5 frequency-domain features from FFT spectrum.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D raw signal.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    features : ndarray (5,)
        [spectral_centroid, spectral_variance, dominant_freq_energy_ratio,
         spectral_kurtosis, top3_band_energy_ratio]
    """
    sig = signal.astype(np.float64)
    n = len(sig)

    # FFT magnitude
    spectrum = np.abs(rfft(sig))
    freqs = rfftfreq(n, 1.0 / fs)

    # Skip DC component
    spectrum = spectrum[1:]
    freqs = freqs[1:]

    total_energy = np.sum(spectrum**2) + 1e-12

    # Spectral centroid
    spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)

    # Spectral variance
    spectral_variance = np.sum(((freqs - spectral_centroid)**2) * spectrum) / (np.sum(spectrum) + 1e-12)

    # Dominant frequency energy ratio
    dom_idx = np.argmax(spectrum)
    dominant_freq_energy_ratio = spectrum[dom_idx]**2 / total_energy

    # Spectral kurtosis
    spec_kurt = sp_kurtosis(spectrum)

    # Top-3 frequency band energy ratio
    band_width = fs / (2 * 3)
    band_energies = np.zeros(3)
    for b in range(3):
        lo = b * band_width
        hi = (b + 1) * band_width
        mask = (freqs >= lo) & (freqs < hi)
        band_energies[b] = np.sum(spectrum[mask]**2)
    top3_band_energy_ratio = np.max(band_energies) / (total_energy + 1e-12)

    return np.array([
        spectral_centroid, spectral_variance, dominant_freq_energy_ratio,
        spec_kurt, top3_band_energy_ratio
    ])


def extract_wpt_features(signal, fs, wavelet="db4", max_level=4):
    """
    Extract WPT leaf node normalized energies as features.

    Reference:
      https://github.com/Western-OC2-Lab/Vibration-Based-Fault-Diagnosis-with-Low-Delay

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D signal.
    fs : float
        Sampling rate (unused, kept for interface consistency).
    wavelet : str
        Wavelet name.
    max_level : int
        WPT decomposition level.

    Returns
    -------
    features : ndarray (2**max_level,)
        Normalized energy per leaf node.
    """
    import pywt
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="symmetric",
                            maxlevel=max_level)
    nodes = wp.get_level(max_level, "natural")
    n_nodes = len(nodes)
    energies = np.zeros(n_nodes)
    for i, node in enumerate(nodes):
        energies[i] = np.sum(node.data**2)
    total = np.sum(energies)
    if total > 1e-12:
        energies = energies / total
    return energies


def extract_stft_features(signal, fs, nperseg=256, noverlap=192, nfft=None,
                          n_freq_bands=32):
    """
    Extract STFT band normalized energies as features.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D signal.
    fs : float
        Sampling rate.
    nperseg : int
    noverlap : int
    nfft : int or None
    n_freq_bands : int
        Number of frequency bands to aggregate into.

    Returns
    -------
    features : ndarray (n_freq_bands,)
        Normalized energy per frequency band.
    """
    if nfft is None:
        nfft = nperseg
    from scipy.signal import stft
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Zxx_mag = np.abs(Zxx)  # (n_freq, n_time)

    # Aggregate into n_freq_bands equal-width bands
    n_freq_total = Zxx_mag.shape[0]
    band_energies = np.zeros(n_freq_bands)
    for b in range(n_freq_bands):
        lo = int(b * n_freq_total / n_freq_bands)
        hi = int((b + 1) * n_freq_total / n_freq_bands)
        band_energies[b] = np.mean(Zxx_mag[lo:hi]**2)
    total = np.sum(band_energies)
    if total > 1e-12:
        band_energies = band_energies / total
    return band_energies


def extract_features(signal, fs, feature_set="all", **kwargs):
    """
    Unified feature extraction entry point.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D signal.
    fs : float
        Sampling rate in Hz.
    feature_set : str
        "all" — time (10) + freq (5) + time-freq (WPT + STFT)
        "time" — time-domain features only (10-dim)
        "freq" — frequency-domain features only (5-dim)
        "time_freq" — WPT + STFT features only (variable-dim)
    **kwargs :
        Passed to sub-extractors.

    Returns
    -------
    features : ndarray (n_features,)
        Concatenated feature vector.
    """
    feature_set = feature_set.lower()
    parts = []

    if feature_set in ("all", "time"):
        parts.append(extract_time_domain_features(signal))

    if feature_set in ("all", "freq"):
        parts.append(extract_freq_domain_features(signal, fs))

    if feature_set in ("all", "time_freq"):
        parts.append(extract_wpt_features(signal, fs, **kwargs))
        parts.append(extract_stft_features(signal, fs, **kwargs))

    if not parts:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return np.concatenate(parts)


# Aliases for convenience
extract_time_features = extract_time_domain_features
extract_freq_features = extract_freq_domain_features
