"""
NMF (Non-negative Matrix Factorization) 模块
"""

import numpy as np
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def run_nmf(
    X: np.ndarray,
    n_sources: Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-negative Matrix Factorization via sklearn.

    Note: Only applicable when X is non-negative (e.g. STFT magnitude spectra).
    If X contains negative values, it will be shifted to be non-negative.

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix (should be non-negative).
    n_sources : int or None
        Number of components.
    max_iter : int
    tol : float
    random_state : int

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
    A_est : ndarray (n_obs, n_sources) — mixing matrix: X ≈ A @ S
    W : ndarray (n_sources, n_obs) — pseudo-demixing matrix (pinv(A))
    """
    from sklearn.decomposition import NMF

    n_obs, T = X.shape
    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    X_wrk = X.copy()
    # Ensure non-negative: shift by minimum value per observation
    for i in range(n_obs):
        mn = X_wrk[i].min()
        if mn < 0:
            X_wrk[i] = X_wrk[i] - mn

    nmf = NMF(n_components=n_sources, max_iter=max_iter, tol=tol,
              random_state=random_state, **kwargs)
    nmf.fit(X_wrk.T)
    # nmf.components_ is H: (n_sources, n_obs)
    # nmf.transform returns W: (n_samples, n_sources)
    A_est = nmf.components_.T  # (n_obs, n_sources): mixing
    S_est = nmf.transform(X_wrk.T).T  # (n_sources, n_samples): sources

    # Pseudo-demixing via pseudoinverse
    try:
        W = np.linalg.pinv(A_est)  # (n_sources, n_obs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A_est, np.eye(n_obs))[0].T

    return S_est, A_est, W
