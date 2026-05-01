"""
SOBI (Second Order Blind Identification) 算法
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def run_sobi(
    X: np.ndarray,
    n_sources: Optional[int] = None,
    n_lags: int = 50,
    max_iter: int = 100,
    tol: float = 1e-6,
    use_all_lags: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SOBI (Second Order Blind Identification).

    Steps:
    1. Centering
    2. Whitening via eigenvalue decomposition
    3. Compute time-lagged covariance matrices R(τ_k)
    4. Symmetrize each R(τ_k)
    5. Joint diagonalization (Jacobi-like)
    6. Compute demixing matrix and sources

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix (mixed signals).
    n_sources : int or None
        Number of sources to extract. If None, equals n_obs.
    n_lags : int
        Number of time lags for covariance estimation.
    max_iter : int
        Maximum Jacobi iterations.
    tol : float
        Convergence tolerance for rotation angles.
    use_all_lags : bool
        If True, use lags 1..n_lags. If False, skip first few.

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
        Estimated source signals.
    W : ndarray (n_sources, n_obs)
        Demixing matrix: S = W @ X.
    """
    n_obs, T = X.shape

    if n_sources is None:
        n_sources = n_obs
    n_sources = min(n_sources, n_obs)

    # Step 1: Centering
    X_centered = X - np.mean(X, axis=1, keepdims=True)

    # Step 2: Whitening
    # Covariance matrix R(0) = X @ X.T / T
    R0 = X_centered @ X_centered.T / T
    R0 = (R0 + R0.T) / 2  # Ensure symmetry

    # Eigenvalue decomposition
    D, E = eigh(R0)
    # Sort descending
    idx = np.argsort(D)[::-1]
    D = D[idx]
    E = E[:, idx]

    # Select top n_sources dimensions
    D_top = np.maximum(D[:n_sources], 1e-12)
    E_top = E[:, :n_sources]

    # Whitening matrix
    V = np.diag(1.0 / np.sqrt(D_top)) @ E_top.T  # shape (n_sources, n_obs)
    Z = V @ X_centered  # shape (n_sources, T), whitened data

    # Step 3: Compute time-lagged covariance matrices
    lag_start = 1 if use_all_lags else max(1, n_sources)
    lags = np.arange(lag_start, lag_start + n_lags)
    R_lags = []

    for tau in lags:
        if tau >= T - 1:
            continue
        # R_z(τ) = Z[:, τ:] @ Z[:, :T-τ].T / (T - τ)
        R_tau = Z[:, tau:] @ Z[:, :T - tau].T / (T - tau)
        # Symmetrize
        R_tau_sym = (R_tau + R_tau.T) / 2
        R_lags.append(R_tau_sym)

    if len(R_lags) == 0:
        raise ValueError("Too few samples for the requested n_lags. Reduce n_lags.")

    # Step 4: Joint diagonalization (Jacobi-like)
    U = _joint_diagonalize_jacobi(R_lags, max_iter, tol)

    # Step 5: Demixing matrix
    W = U.T @ V  # shape (n_sources, n_obs)
    S_est = W @ X_centered  # shape (n_sources, T)

    return S_est, W


def _joint_diagonalize_jacobi(
    R_list: list,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Joint diagonalization of a set of symmetric matrices using Jacobi-like
    algorithm (Cardoso & Souloumiac 1996).

    Parameters
    ----------
    R_list : list of ndarray
        List of K symmetric (n, n) matrices to jointly diagonalize.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on rotation angles.

    Returns
    -------
    U : ndarray (n, n)
        Orthogonal matrix such that U @ R_k @ U.T are approximately diagonal.
    """
    K = len(R_list)
    n = R_list[0].shape[0]
    U = np.eye(n)

    for iteration in range(max_iter):
        max_angle = 0.0

        for i in range(n - 1):
            for j in range(i + 1, n):
                # Extract relevant entries from all R_k
                # x_k = R_k[i, j]
                # y_k = (R_k[i, i] - R_k[j, j]) / 2
                x = np.zeros(K)
                y = np.zeros(K)
                for kk in range(K):
                    x[kk] = R_list[kk][i, j]
                    y[kk] = (R_list[kk][i, i] - R_list[kk][j, j]) / 2.0

                # Compute sums
                S_xy = np.dot(x, y)
                S_xx = np.dot(x, x)
                S_yy = np.dot(y, y)

                # tan(4θ) = 2*S_xy / (S_yy - S_xx)
                # θ = 0.25 * atan2(2*S_xy, S_yy - S_xx)
                angle = 0.25 * np.arctan2(2.0 * S_xy, S_yy - S_xx)

                if abs(angle) > max_angle:
                    max_angle = abs(angle)

                if abs(angle) < tol / 2:
                    continue

                # Givens rotation matrix G
                c = np.cos(angle)
                s = np.sin(angle)

                # Update U: U_new = G @ U
                U_i = U[i, :].copy()
                U_j = U[j, :].copy()
                U[i, :] = c * U_i - s * U_j
                U[j, :] = s * U_i + c * U_j

                # Update all R_k: R_k_new = G @ R_k @ G^T
                for kk in range(K):
                    R = R_list[kk]
                    Ri = R[i, :].copy()
                    Rj = R[j, :].copy()

                    # G @ R: update rows i and j
                    R[i, :] = c * Ri - s * Rj
                    R[j, :] = s * Ri + c * Rj

                    # (G @ R) @ G^T: update columns i and j
                    Ci = R[:, i].copy()  # note: after row update
                    Cj = R[:, j].copy()
                    R[:, i] = c * Ci - s * Cj
                    R[:, j] = s * Ci + c * Cj

        if max_angle < tol:
            break

    return U
