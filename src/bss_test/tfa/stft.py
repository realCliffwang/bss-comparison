"""
短时傅里叶变换 (STFT) 模块
"""

import numpy as np
from scipy.signal import stft as sp_signal_stft
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def stft_transform(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = 192,
    nfft: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    n_bands: int = 20,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform via scipy.signal.stft.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of overlap samples.
    nfft : int or None
        FFT length (defaults to nperseg).
    freq_range : tuple (low, high) or None
        Frequency range to extract (Hz). If None, uses full range.
    n_bands : int
        Number of frequency bands to return (selects top-energy bands if
        fewer than total STFT bins).

    Returns
    -------
    matrix : ndarray (n_bands, n_samples)
        STFT magnitude (absolute value), interpolated to match input length.
    freq_axes : ndarray (n_bands,)
        Frequency values for each output band.
    """
    if nfft is None:
        nfft = nperseg

    f, t_stft, Zxx = sp_signal_stft(signal, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap, nfft=nfft, **kwargs)
    Zxx_mag = np.abs(Zxx)  # shape (n_freq, n_time)

    # Select frequency range
    if freq_range is not None:
        f_low, f_high = freq_range
        mask = (f >= f_low) & (f <= f_high)
        Zxx_mag = Zxx_mag[mask]
        f = f[mask]

    n_freq_total = Zxx_mag.shape[0]

    # If more bins than n_bands, select top-energy bands
    if n_freq_total > n_bands:
        band_energy = np.mean(Zxx_mag**2, axis=1)
        top_idx = np.argsort(band_energy)[-n_bands:]
        top_idx = np.sort(top_idx)
        Zxx_mag = Zxx_mag[top_idx]
        f = f[top_idx]

    n_time_stft = Zxx_mag.shape[1]
    n_samples = len(signal)

    # Interpolate to match original signal length
    if n_time_stft != n_samples:
        t_orig = np.linspace(0, n_samples - 1, n_samples)
        t_stft_idx = np.linspace(0, n_samples - 1, n_time_stft)
        Zxx_interp = np.zeros((Zxx_mag.shape[0], n_samples))
        for i in range(Zxx_mag.shape[0]):
            Zxx_interp[i] = np.interp(t_orig, t_stft_idx, Zxx_mag[i])
        matrix = Zxx_interp
    else:
        matrix = Zxx_mag

    return matrix, f
