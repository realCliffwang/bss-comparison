"""
PCA (Principal Component Analysis) 模块
"""

import numpy as np
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def run_pca(
    X: np.ndarray,
    n_sources: Optional[int] = None,
    whiten: bool = False,
    random_state: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal Component Analysis via sklearn (baseline).

    This is the weakest separation baseline; BSS methods should outperform it
    when latent sources are non-Gaussian or have temporal structure.

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix.
    n_sources : int or None
        Number of components to keep.
    whiten : bool
        If True, whiten the output so components have unit variance.

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
    A_est : ndarray (n_obs, n_sources) — mixing matrix (components_.T)
    W : ndarray (n_sources, n_obs) — demixing matrix (components_)
    """
    from sklearn.decomposition import PCA

    n_obs, T = X.shape
    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    pca = PCA(n_components=n_sources, whiten=whiten,
              random_state=random_state, **kwargs)
    S_est = pca.fit_transform(X.T).T  # (n_sources, n_samples)
    W = pca.components_  # (n_sources, n_obs)
    A_est = W.T  # (n_obs, n_sources), for consistency

    return S_est, A_est, W
