"""
经验模态分解 (EMD/EEMD/CEEMDAN) 模块
"""

import numpy as np
from scipy.signal import hilbert as sp_hilbert
from typing import Tuple

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def emd_transform(
    signal: np.ndarray,
    fs: float,
    max_imf: int = 6,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical Mode Decomposition via PyEMD.

    Extract first N IMF components as pseudo-channels.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    max_imf : int
        Maximum number of IMFs to extract.

    Returns
    -------
    matrix : ndarray (max_imf, n_samples)
        First N IMF components.
    freq_axes : ndarray (max_imf,)
        Mean instantaneous frequency of each IMF (Hz), via Hilbert transform.
    """
    try:
        from PyEMD import EMD
    except ImportError:
        raise ImportError(
            "EMD requires PyEMD. Install with: pip install EMD-signal\n"
            "GitHub: https://github.com/laszukdawid/PyEMD"
        )

    emd = EMD()
    imfs = emd.emd(signal, max_imf=max_imf)

    n_imfs = min(imfs.shape[0], max_imf)
    if imfs.ndim == 2:
        imfs = imfs[:n_imfs]
    else:
        imfs = imfs.reshape(1, -1)[:n_imfs]

    # Estimate mean frequency of each IMF via Hilbert transform
    mean_freqs = np.zeros(n_imfs)
    for i in range(n_imfs):
        analytic = sp_hilbert(imfs[i])
        phase = np.unwrap(np.angle(analytic))
        if len(phase) > 1:
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            mean_freqs[i] = float(np.mean(np.abs(inst_freq)))
        else:
            mean_freqs[i] = 0.0

    matrix = np.array(imfs, dtype=np.float64)
    return matrix, mean_freqs


def eemd_transform(
    signal: np.ndarray,
    fs: float,
    max_imf: int = 6,
    trials: int = 100,
    noise_std: float = 0.2,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble Empirical Mode Decomposition via PyEMD.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate.
    max_imf : int
        Max IMFs to extract.
    trials : int
        Number of ensemble trials.
    noise_std : float
        Standard deviation of added noise relative to signal std.

    Returns
    -------
    matrix : ndarray (max_imf, n_samples)
    freq_axes : ndarray (max_imf,)
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        raise ImportError(
            "EEMD requires PyEMD. Install with: pip install EMD-signal\n"
            "GitHub: https://github.com/laszukdawid/PyEMD"
        )

    eemd = EEMD(trials=trials, noise_width=noise_std)
    imfs = eemd.eemd(signal, max_imf=max_imf)

    n_imfs = min(imfs.shape[0], max_imf)
    if imfs.ndim == 2:
        imfs = imfs[:n_imfs]
    else:
        imfs = imfs.reshape(1, -1)[:n_imfs]

    mean_freqs = np.zeros(n_imfs)
    for i in range(n_imfs):
        analytic = sp_hilbert(imfs[i])
        phase = np.unwrap(np.angle(analytic))
        if len(phase) > 1:
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            mean_freqs[i] = float(np.mean(np.abs(inst_freq)))
        else:
            mean_freqs[i] = 0.0

    matrix = np.array(imfs, dtype=np.float64)
    return matrix, mean_freqs


def ceemdan_transform(
    signal: np.ndarray,
    fs: float,
    max_imf: int = 6,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete Ensemble EMD with Adaptive Noise via PyEMD.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate.
    max_imf : int
        Max IMFs to extract.

    Returns
    -------
    matrix : ndarray (max_imf, n_samples)
    freq_axes : ndarray (max_imf,)
    """
    try:
        from PyEMD import CEEMDAN
    except ImportError:
        raise ImportError(
            "CEEMDAN requires PyEMD. Install with: pip install EMD-signal\n"
            "GitHub: https://github.com/laszukdawid/PyEMD"
        )

    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(signal, max_imf=max_imf)

    n_imfs = min(imfs.shape[0], max_imf)
    if imfs.ndim == 2:
        imfs = imfs[:n_imfs]
    else:
        imfs = imfs.reshape(1, -1)[:n_imfs]

    mean_freqs = np.zeros(n_imfs)
    for i in range(n_imfs):
        analytic = sp_hilbert(imfs[i])
        phase = np.unwrap(np.angle(analytic))
        if len(phase) > 1:
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            mean_freqs[i] = float(np.mean(np.abs(inst_freq)))
        else:
            mean_freqs[i] = 0.0

    matrix = np.array(imfs, dtype=np.float64)
    return matrix, mean_freqs


def emd_factory(
    signal: np.ndarray,
    fs: float,
    method: str = "emd",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EMD 方法工厂函数

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate.
    method : str
        EMD 方法: "emd", "eemd", "ceemdan"
    **kwargs : dict
        方法特定参数

    Returns
    -------
    matrix : ndarray (n_imf, n_samples)
    freq_axes : ndarray (n_imf,)
    """
    method = method.lower()
    
    if method == "emd":
        return emd_transform(signal, fs, **kwargs)
    elif method == "eemd":
        return eemd_transform(signal, fs, **kwargs)
    elif method == "ceemdan":
        return ceemdan_transform(signal, fs, **kwargs)
    else:
        raise ValueError(f"未知的 EMD 方法: {method}")
