"""
ICA (Independent Component Analysis) 模块
包含 FastICA 和 PICARD 算法
"""

import numpy as np
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def run_fastica(
    X: np.ndarray,
    n_sources: Optional[int] = None,
    max_iter: int = 2000,
    random_state: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FastICA via sklearn.

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix.
    n_sources : int or None
    max_iter : int
    random_state : int

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
    W : ndarray (n_sources, n_obs)
        Demixing matrix (unmixing_matrix_ from sklearn).
    """
    from sklearn.decomposition import FastICA

    n_obs, T = X.shape
    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    ica = FastICA(n_components=n_sources, max_iter=max_iter,
                  random_state=random_state, **kwargs)
    # FastICA expects shape (n_samples, n_features), so transpose
    S_est = ica.fit_transform(X.T).T  # (n_sources, n_samples)
    W = ica.components_  # (n_sources, n_obs) — this is the unmixing matrix

    return S_est, W


def run_picard(
    X: np.ndarray,
    n_sources: Optional[int] = None,
    max_iter: int = 500,
    tol: float = 1e-7,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PICARD (Preconditioned ICA for Real Data).

    Reference: Ablin, Gramfort, Cardoso, Bach (2018)
    "Faster ICA under orthogonal constraint." ICASSP.

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix.
    n_sources : int or None
        Number of sources.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
    A_est : ndarray (n_obs, n_sources)
    W : ndarray (n_sources, n_obs)
    """
    try:
        import picard
    except ImportError:
        raise ImportError(
            "PICARD requires python-picard. Install with:\n"
            "  pip install python-picard\n"
            "GitHub: https://github.com/pierreablin/picard"
        )

    n_obs, T = X.shape
    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    # picard expects (n_samples, n_features).
    # In BSS: X is (n_obs, n_samples). Pass X as-is so picard treats
    # n_obs as "samples" and n_samples as "features", reducing n_obs to
    # n_sources components. Output S will be (n_sources, n_samples).
    max_T = 20000
    X_trunc = X[:, :max_T] if X.shape[1] > max_T else X
    X_trunc = X_trunc.astype(np.float64)  # (n_obs, T_trunc)

    K, W_ica, S = picard.picard(
        X_trunc, n_components=n_sources, max_iter=max_iter, tol=tol,
        ortho=True, whiten=True
    )
    # S = W_ica @ K @ X_trunc → shape (n_sources, T_trunc)
    # K: (n_sources, n_obs), W_ica: (n_sources, n_sources)
    S_est = np.asarray(S, dtype=np.float64)  # (n_sources, T_trunc)
    # Demixing: D = W_ica @ K → (n_sources, n_obs), S_est = D @ X
    W_full = np.asarray(W_ica @ K, dtype=np.float64)  # (n_sources, n_obs)
    A_est = np.linalg.pinv(W_full)

    # Reconstruct full-length sources if X was truncated
    if X.shape[1] > max_T:
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        S_est = W_full @ X_centered  # (n_sources, T)

    return S_est, A_est, W_full
