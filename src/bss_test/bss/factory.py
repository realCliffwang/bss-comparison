"""
盲源分离工厂模块
"""

import numpy as np
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def bss_factory(
    X: np.ndarray,
    method: str = "sobi",
    n_components: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一 BSS 工厂函数

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        观测矩阵
    method : str
        BSS 方法: "sobi", "fastica", "jade", "picard", "nmf", "pca"
    n_components : int or None
        源数量
    **kwargs : dict
        方法特定参数

    Returns
    -------
    S_est : ndarray (n_components, n_samples)
        估计源信号
    A_est : ndarray (n_obs, n_components)
        估计混合矩阵
    W : ndarray (n_components, n_obs)
        分离矩阵
    """
    if n_components is None:
        n_components = X.shape[0]
    
    method = method.lower()
    
    if method == "sobi":
        from bss_test.bss.sobi import run_sobi
        S_est, W = run_sobi(X, n_sources=n_components, **kwargs)
        A_est = np.linalg.pinv(W)
        return S_est, A_est, W
    
    elif method == "fastica":
        from bss_test.bss.ica import run_fastica
        S_est, W = run_fastica(X, n_sources=n_components, **kwargs)
        A_est = np.linalg.pinv(W)
        return S_est, A_est, W
    
    elif method == "jade":
        from bss_test.bss.jade import run_jade
        return run_jade(X, n_sources=n_components, **kwargs)
    
    elif method == "picard":
        from bss_test.bss.ica import run_picard
        return run_picard(X, n_sources=n_components, **kwargs)
    
    elif method == "nmf":
        from bss_test.bss.nmf import run_nmf
        return run_nmf(X, n_sources=n_components, **kwargs)
    
    elif method == "pca":
        from bss_test.bss.pca import run_pca
        return run_pca(X, n_sources=n_components, **kwargs)
    
    else:
        raise ValueError(
            f"未知的 BSS 方法: {method}. "
            f"支持的方法: sobi, fastica, jade, picard, nmf, pca"
        )


def run_bss(
    X_for_bss: np.ndarray,
    method: str = "SOBI",
    n_sources: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一 BSS 接口（兼容旧代码）

    Parameters
    ----------
    X_for_bss : ndarray (n_obs, n_samples)
        观测矩阵
    method : str
        "SOBI" | "FastICA" | "JADE" | "PICARD" | "NMF" | "PCA"
    n_sources : int or None
        源数量
    **kwargs : dict
        传递给具体 BSS 函数的参数

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
        估计源信号
    W : ndarray (n_sources, n_obs) or None
        分离矩阵
    """
    sources, A_est, W = bss_factory(X_for_bss, method, n_components=n_sources,
                                     **kwargs)
    return sources, W
