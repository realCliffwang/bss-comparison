"""
Blind Source Separation (BSS) module.

Methods implemented:
1. SOBI (Second Order Blind Identification) — self-implemented
   Reference: Belouchrani et al., "A blind source separation technique using
   second-order statistics," IEEE Trans. Signal Processing, 1997.
   Joint diagonalization: Cardoso & Souloumiac, "Jacobi angles for simultaneous
   diagonalization," SIAM J. Matrix Anal. Appl., 1996.

2. FastICA — wrapper for sklearn.decomposition.FastICA
   Reference: Hyvarinen & Oja, "Independent Component Analysis: Algorithms and
   Applications," Neural Networks, 2000.

3. JADE — via jadeR (Cardoso 1999), copied to src/third_party/jadeR.py
   Reference: Cardoso, J. (1999) "High-order contrasts for independent
   component analysis." Neural Computation, 11(1): 157-192.

4. PICARD — via python-picard (Ablin et al. 2018)
   Reference: Ablin, Gramfort, Cardoso, Bach (2018) "Faster ICA under
   orthogonal constraint." ICASSP.

5. NMF — via sklearn.decomposition.NMF
   Non-negative matrix factorization (requires non-negative input, e.g.
   STFT magnitude spectra).

6. PCA — via sklearn.decomposition.PCA
   Principal component analysis (weakest baseline, demonstrates BSS necessity).
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh


def run_sobi(X, n_sources=None, n_lags=50, max_iter=100, tol=1e-6,
             use_all_lags=True):
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


def _joint_diagonalize_jacobi(R_list, max_iter=100, tol=1e-6):
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


def run_fastica(X, n_sources=None, max_iter=2000, random_state=42, **kwargs):
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


def run_bss(X_for_bss, method="SOBI", n_sources=None, **kwargs):
    """
    Unified interface for BSS.

    Parameters
    ----------
    X_for_bss : ndarray (n_obs, n_samples)
        Observation matrix.
    method : str
        "SOBI" | "FastICA" | "JADE" | "PICARD" | "NMF" | "PCA"
    n_sources : int or None
        Number of sources to estimate.
    **kwargs : dict
        Passed to the specific BSS function.

    Returns
    -------
    S_est : ndarray (n_sources, n_samples)
        Estimated sources.
    W : ndarray (n_sources, n_obs) or None
        Demixing matrix (SOBI/JADE/PICARD: yes, FastICA: unmixing matrix,
        NMF: pinv of mixing, PCA: components_).
    """
    sources, A_est, W = bss_factory(X_for_bss, method, n_components=n_sources,
                                     **kwargs)
    return sources, W


def run_jade(X, n_sources=None, **kwargs):
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
    _THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "third_party")
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


def run_picard(X, n_sources=None, max_iter=500, tol=1e-7, **kwargs):
    """
    PICARD (Preconditioned ICA for Real Data).

    GitHub ref: https://github.com/pierreablin/picard
    Install: pip install python-picard

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


def run_nmf(X, n_sources=None, max_iter=1000, tol=1e-4, random_state=42,
            **kwargs):
    """
    Non-negative Matrix Factorization via sklearn.

    GitHub ref: https://github.com/scikit-learn/scikit-learn

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


def run_pca(X, n_sources=None, whiten=False, random_state=42, **kwargs):
    """
    Principal Component Analysis via sklearn (baseline).

    GitHub ref: https://github.com/scikit-learn/scikit-learn

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


def bss_factory(X, method, n_components=None, **kwargs):
    """
    Unified BSS factory.

    Parameters
    ----------
    X : ndarray (n_obs, n_samples)
        Observation matrix.
    method : str
        "sobi" | "fastica" | "jade" | "picard" | "nmf" | "pca"
    n_components : int or None
        Number of sources/components.
    **kwargs :
        Passed to the specific BSS function.

    Returns
    -------
    sources : ndarray (n_sources, n_samples)
        Estimated source signals.
    mixing_matrix : ndarray (n_obs, n_sources)
        X ≈ mixing_matrix @ sources
    unmixing_matrix : ndarray (n_sources, n_obs)
        sources = unmixing_matrix @ X
    """
    method = method.lower()
    if method == "sobi":
        S_est, W = run_sobi(X, n_sources=n_components, **kwargs)
        A_est = np.linalg.pinv(W)
        return S_est, A_est, W
    elif method == "fastica":
        S_est, W = run_fastica(X, n_sources=n_components, **kwargs)
        A_est = np.linalg.pinv(W)
        return S_est, A_est, W
    elif method == "jade":
        return run_jade(X, n_sources=n_components, **kwargs)
    elif method == "picard":
        return run_picard(X, n_sources=n_components, **kwargs)
    elif method == "nmf":
        return run_nmf(X, n_sources=n_components, **kwargs)
    elif method == "pca":
        return run_pca(X, n_sources=n_components, **kwargs)
    else:
        raise ValueError(
            f"Unknown BSS method: {method}. "
            f"Supported: sobi, fastica, jade, picard, nmf, pca"
        )
