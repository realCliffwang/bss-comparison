"""
JADE (Joint Approximate Diagonalization of Eigenmatrices) 算法
"""

import sys
import os
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

    Uses jadeR from src/third_party/jadeR.py (Cardoso 1999).
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
    # Import jadeR from third_party
    _THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "third_party")
    if _THIRD_PARTY not in sys.path:
        sys.path.insert(0, _THIRD_PARTY)
    from jadeR import jadeR as _jadeR

    n_obs, T = X.shape
    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    W = _jadeR(X, m=n_sources, verbose=kwargs.get("verbose", False))  # (m, n)
    W = np.asarray(W, dtype=np.float64)
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    S_est = W @ X_centered  # (n_sources, n_samples)
    A_est = np.linalg.pinv(W)  # (n_obs, n_sources)

    return S_est, A_est, W
