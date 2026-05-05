"""
BSS evaluation metrics.

Provides quantitative metrics for assessing blind source separation quality:
- SIR (Signal-to-Interference Ratio) for synthetic mixtures
- Source independence metric (off-diagonal correlation)
- Fault Frequency Detection Score (FFDS) for bearing diagnostics
"""

import numpy as np
from scipy.signal import hilbert

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(S_true, S_est):
    """
    Compute SIR and correlation for synthetic mixture evaluation.

    Parameters
    ----------
    S_true : ndarray (n_sources, n_samples)
        Ground truth sources.
    S_est : ndarray (n_sources, n_samples)
        Estimated sources.

    Returns
    -------
    metrics : dict
        {"SIR_dB": float, "mean_correlation": float}
    """
    n_src = min(S_true.shape[0], S_est.shape[0])
    S_true_norm = S_true[:n_src] / (np.std(S_true[:n_src], axis=1, keepdims=True) + 1e-12)
    S_est_norm = S_est[:n_src] / (np.std(S_est[:n_src], axis=1, keepdims=True) + 1e-12)

    corr_matrix = np.abs(np.corrcoef(S_true_norm, S_est_norm)[:n_src, n_src:])
    best_corrs = []
    used = set()
    for _ in range(n_src):
        best_val = -1
        best_pair = (-1, -1)
        for ii in range(corr_matrix.shape[0]):
            for jj in range(corr_matrix.shape[1]):
                if jj not in used and corr_matrix[ii, jj] > best_val:
                    best_val = corr_matrix[ii, jj]
                    best_pair = (jj,)
        if best_val > -1:
            used.add(best_pair[0])
            best_corrs.append(best_val)

    mean_corr = np.mean(best_corrs) if best_corrs else 0.0

    errors = []
    for i in range(n_src):
        source_energy = np.var(S_true_norm[i])
        residual = S_true_norm[i] - S_est_norm[i]
        noise_energy = np.var(residual)
        if noise_energy > 1e-12:
            errors.append(10 * np.log10(source_energy / noise_energy))
    mean_sir = np.mean(errors) if errors else float("inf")

    return {"SIR_dB": mean_sir, "mean_correlation": mean_corr}


def compute_independence_metric(S_est):
    """
    Compute source independence: mean absolute off-diagonal correlation.

    Lower = sources are more independent = better separation.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)

    Returns
    -------
    float : mean |off-diagonal correlation|, 0 = perfect independence
    """
    n_src = S_est.shape[0]
    if n_src < 2:
        return 0.0
    corr = np.corrcoef(S_est)
    mask = ~np.eye(n_src, dtype=bool)
    off_diag = np.abs(corr[mask])
    return float(np.mean(off_diag))


def compute_fault_detection_score(S_est, fs, fault_freqs, tol_hz=5.0):
    """
    Fault Frequency Detection Score (FFDS):
    mean(peak at each fault freq) / median noise floor of envelope spectrum.

    Higher = fault frequency more prominent = better diagnostic value.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)
    fs : float
    fault_freqs : dict {name: freq_hz}
    tol_hz : float

    Returns
    -------
    float : FFDS score
    """
    sig = S_est[0]
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    N = len(envelope)
    env_spec = np.abs(np.fft.rfft(envelope))
    freq = np.fft.rfftfreq(N, 1.0 / fs)

    mask_noise = (freq >= 20) & (freq <= 500)
    if np.sum(mask_noise) < 10:
        return 0.0
    noise_floor = np.median(env_spec[mask_noise]) + 1e-12

    peaks = []
    for ff in fault_freqs.values():
        mask = (freq >= ff - tol_hz) & (freq <= ff + tol_hz)
        if np.sum(mask) > 0:
            peaks.append(float(np.max(env_spec[mask])))
    if not peaks:
        return 0.0

    return float(np.mean(peaks)) / noise_floor


def evaluate_bss(S_est, W, X_original, fs, config=None):
    """
    Main evaluation entry point.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)
        Estimated sources.
    W : ndarray (n_sources, n_obs)
        Demixing matrix.
    X_original : ndarray (n_obs, n_samples)
        Original observation (mixed) signals.
    fs : float
        Sampling rate.
    config : dict
        Additional configuration.
    """
    logger.info(f"\n{'='*60}")
    logger.info("BSS Evaluation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Observations: {X_original.shape[0]}, Sources: {S_est.shape[0]}")
    logger.info(f"  Samples: {S_est.shape[1]}, Sampling rate: {fs} Hz")
    logger.info(f"  Duration: {S_est.shape[1]/fs:.2f} s")
    logger.info(f"{'='*60}\n")
