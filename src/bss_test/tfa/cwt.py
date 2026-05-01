"""
连续小波变换 (CWT) 模块
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def cwt_transform(
    signal: np.ndarray,
    fs: float,
    wavelet: str = "cmor1.5-1.0",
    n_bands: int = 20,
    freq_range: Optional[Tuple[float, float]] = None,
    max_scale: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1D Continuous Wavelet Transform.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Wavelet type. Recommended:
        - "cmor1.5-1.0": Complex Morlet, good for time-frequency analysis
        - "morl": Real Morlet
        - "cgau8": Complex Gaussian
    freq_range : tuple (low, high) or None
        Frequency range in Hz. If None, uses (20, fs/2).
    n_bands : int
        Number of frequency bands (scales).
    max_scale : int or None
        Maximum scale. If None, auto-computed.

    Returns
    -------
    coef : ndarray (n_bands, n_samples)
        CWT coefficient magnitudes (absolute value).
    freqs : ndarray (n_bands,)
        Frequencies corresponding to each band.
    scales : ndarray (n_bands,)
        Scales used in CWT.
    """
    N = len(signal)

    if freq_range is None:
        f_low = 20.0
        f_high = fs / 2.0
    else:
        f_low, f_high = freq_range
        f_low = max(f_low, 0.1)
        f_high = min(f_high, fs / 2.0)

    # Convert frequency range to scale range
    # For Morlet-type wavelets: scale = center_freq * fs / (f * sampling_period)
    # pywt.cwt uses: frequency = center_frequency / (scale * sampling_period)
    # where sampling_period = 1/fs
    center_freq = pywt.scale2frequency(wavelet, 1)
    # frequency = center_freq / (scale * dt) = center_freq * fs / scale
    # => scale = center_freq * fs / frequency

    scale_low = center_freq * fs / f_high    # smaller frequency -> larger scale
    scale_high = center_freq * fs / f_low

    # Logarithmic spacing for scales (better frequency resolution at low freq)
    scales = np.logspace(np.log10(scale_low), np.log10(scale_high), n_bands)

    if max_scale is not None:
        scales = np.minimum(scales, max_scale)

    # Handle edge case where scales may be rounded to 1
    scales = np.maximum(scales, 1.0)
    scales = np.unique(scales.astype(int)).astype(float)

    # pywt.cwt returns: coef shape (n_scales, n_samples)
    # Note: pywt 1.4+ API: pywt.cwt(signal, scales, wavelet)
    # Older API: pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    try:
        coef, _freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs)
    except TypeError:
        # Fallback for older pywt versions
        coef, _freqs = pywt.cwt(signal, scales, wavelet)

    # Take magnitude (for complex wavelets; for real wavelets abs-value)
    coef_mag = np.abs(coef)

    # Get frequencies array
    if len(_freqs) == len(scales):
        freqs = _freqs
    else:
        freqs = center_freq * fs / scales

    return coef_mag, freqs, scales


def cwt_transform_multichannel(
    signals: np.ndarray,
    fs: float,
    **kwargs,
) -> list:
    """
    Apply CWT to each channel of a multi-channel signal.

    Parameters
    ----------
    signals : ndarray (n_channels, n_samples)
        Multi-channel signal.
    fs : float
        Sampling rate.
    **kwargs : dict
        Passed to cwt_transform().

    Returns
    -------
    cwt_results : list of dict
        Each dict contains: coef, freqs, scales, channel_idx.
    """
    n_channels = signals.shape[0]
    results = []
    for ch in range(n_channels):
        coef, freqs, scales = cwt_transform(signals[ch], fs, **kwargs)
        results.append({
            "coef": coef,
            "freqs": freqs,
            "scales": scales,
            "channel_idx": ch,
        })
    return results


def plot_cwt_spectrogram(
    coef: np.ndarray,
    freqs: np.ndarray,
    n_samples: int,
    fs: float,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot CWT spectrogram (time-frequency representation).

    Parameters
    ----------
    coef : ndarray (n_scales, n_samples)
        CWT coefficient matrix.
    freqs : ndarray (n_scales,)
        Frequencies in Hz.
    n_samples : int
        Number of time samples.
    fs : float
        Sampling rate in Hz.
    title : str or None
    ax : matplotlib axis or None
    vmin, vmax : float or None
        Color scale limits.
    cmap : str
        Colormap name.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    time_axis = np.arange(n_samples) / fs
    extent = [time_axis[0], time_axis[-1], freqs[0], freqs[-1]]

    # Normalize for display
    coef_db = 20 * np.log10(coef / (coef.max(axis=1, keepdims=True) + 1e-12) + 1e-12)

    im = ax.imshow(coef_db, aspect="auto", origin="lower", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude [dB]")
    return fig, ax
