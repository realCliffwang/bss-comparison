"""
Signal preprocessing: detrend, bandpass filter, normalize.

Provides functions for preprocessing vibration signals before analysis.

Usage:
    from src.preprocessing import preprocess_signals
    signals_pre, fs_out = preprocess_signals(signals, fs, config)
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, filtfilt, detrend as sp_detrend

from src.logger import get_logger
from src.exceptions import PreprocessingError, FilterError, NormalizationError

logger = get_logger(__name__)


def preprocess_signals(
    signals: np.ndarray,
    fs: float,
    config: Optional[Dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess multi-channel signals.

    Parameters
    ----------
    signals : ndarray (n_channels, n_samples)
        Raw signals.
    fs : float
        Sampling rate in Hz.
    config : dict or None
        Configuration dictionary with keys:
        - detrend : bool (default True) — remove linear trend
        - bandpass : (lowcut, highcut) or None — apply bandpass filter
        - normalize : str or None — "zscore", "minmax", or None
        - resample_fs : float or None — resample to this rate
        - filter_order : int (default 4) — Butterworth filter order

    Returns
    -------
    signals_pre : ndarray (n_channels, n_samples_pre)
        Preprocessed signals.
    fs_out : float
        Final sampling rate (may differ from fs if resampled).

    Raises
    ------
    PreprocessingError
        If preprocessing fails.
    """
    logger.debug(f"Preprocessing signals: shape={signals.shape}, fs={fs} Hz")

    if config is None:
        config = {}

    detrend_opt = config.get("detrend", True)
    bandpass = config.get("bandpass", None)
    normalize_opt = config.get("normalize", "zscore")
    resample_fs = config.get("resample_fs", None)
    filter_order = config.get("filter_order", 4)

    try:
        sig = signals.copy().astype(np.float64)
    except Exception as e:
        raise PreprocessingError(f"Failed to copy signals: {e}")

    # Detrend
    if detrend_opt:
        logger.debug("Applying detrend")
        for i in range(sig.shape[0]):
            sig[i] = sp_detrend(sig[i])

    # Remove DC
    sig = sig - np.mean(sig, axis=1, keepdims=True)

    # Bandpass filter
    if bandpass is not None:
        lowcut, highcut = bandpass
        logger.debug(f"Applying bandpass filter: {lowcut}-{highcut} Hz, order={filter_order}")
        for i in range(sig.shape[0]):
            try:
                sig[i] = bandpass_filter(sig[i], fs, lowcut, highcut, order=filter_order)
            except Exception as e:
                raise FilterError(
                    f"Bandpass filter failed for channel {i}: {e}",
                    channel=i,
                    lowcut=lowcut,
                    highcut=highcut,
                )

    # Resample
    fs_out = fs
    if resample_fs is not None and resample_fs < fs:
        logger.debug(f"Resampling from {fs} Hz to {resample_fs} Hz")
        from scipy.signal import resample as scipy_resample
        n_new = int(sig.shape[1] * resample_fs / fs)
        resampled = np.zeros((sig.shape[0], n_new))
        for i in range(sig.shape[0]):
            resampled[i] = scipy_resample(sig[i], n_new)
        sig = resampled
        fs_out = resample_fs

    # Normalize
    if normalize_opt == "zscore":
        logger.debug("Applying z-score normalization")
        for i in range(sig.shape[0]):
            std = np.std(sig[i])
            if std > 1e-12:
                sig[i] = sig[i] / std
            else:
                logger.warning(f"Channel {i} has near-zero std, skipping normalization")
    elif normalize_opt == "minmax":
        logger.debug("Applying min-max normalization")
        for i in range(sig.shape[0]):
            s_min, s_max = sig[i].min(), sig[i].max()
            if s_max - s_min > 1e-12:
                sig[i] = (sig[i] - s_min) / (s_max - s_min)
            else:
                logger.warning(f"Channel {i} has near-zero range, skipping normalization")
    elif normalize_opt is None:
        pass
    else:
        raise NormalizationError(f"Unknown normalize option: {normalize_opt}")

    logger.debug(f"Preprocessing complete: output shape={sig.shape}, fs_out={fs_out} Hz")
    return sig, fs_out


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to a 1D signal.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        Input signal.
    fs : float
        Sampling rate in Hz.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered : ndarray (n_samples,)
        Filtered signal.

    Raises
    ------
    FilterError
        If filtering fails.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0 or high >= 1:
        raise FilterError(
            f"Invalid filter frequencies: lowcut={lowcut}, highcut={highcut}, fs={fs}",
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
        )

    try:
        b, a = butter(order, [low, high], btype="band")
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception as e:
        raise FilterError(f"Butterworth filter failed: {e}", order=order)


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """
    Remove linear trend from signal.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        Input signal.

    Returns
    -------
    detrended : ndarray (n_samples,)
        Detrended signal.
    """
    return sp_detrend(signal)


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize signal.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        Input signal.
    method : str
        Normalization method: "zscore", "minmax", or None.

    Returns
    -------
    normalized : ndarray (n_samples,)
        Normalized signal.

    Raises
    ------
    NormalizationError
        If normalization method is unknown.
    """
    if method == "zscore":
        std = np.std(signal)
        if std > 1e-12:
            return signal / std
        return signal
    elif method == "minmax":
        s_min, s_max = signal.min(), signal.max()
        if s_max - s_min > 1e-12:
            return (signal - s_min) / (s_max - s_min)
        return signal
    elif method is None:
        return signal
    else:
        raise NormalizationError(f"Unknown normalization method: {method}")
