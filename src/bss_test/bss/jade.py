"""
JADE (Joint Approximate Diagonalization of Eigenmatrices) 算法
"""

import numpy as np
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def run_jade(
    X: np.ndarray,
    n_sources: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JADE (Joint Approximate Diagonalization of Eigenmatrices).

    Uses jadeR from bss_test.bss._jadeR (Cardoso 1999).
    GitHub ref: https://github.com/gbeckers/jadeR

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix.
    n_sources : int or None
        Number of sources to extract.
    **kwargs :
        verbose : bool — passed to jadeR.

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
    A_est : ndarray (n_obs, n_sources) — mixing matrix: X ≈ A @ S
    W : ndarray (n_sources, n_obs) — demixing matrix: S = W @ X
    """
    from bss_test.bss._jadeR import jadeR as _jadeR

    n_obs, T = X.shape
    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    W = _jadeR(X, m=n_sources, verbose=kwargs.get("verbose", False))
    W = np.asarray(W, dtype=np.float64)
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    S_est = W @ X_centered
    A_est = np.linalg.pinv(W)

    return S_est, A_est, W
