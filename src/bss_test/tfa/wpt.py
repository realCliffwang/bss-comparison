"""
小波包变换 (WPT) 模块
"""

import numpy as np
import pywt
from typing import Tuple

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def wpt_transform(
    signal: np.ndarray,
    fs: float,
    wavelet: str = "db4",
    max_level: int = 4,
    n_bands: int = 16,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wavelet Packet Transform via PyWavelets.

    Extracts leaf-node reconstructed signals as pseudo-channels.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Wavelet name (e.g. "db4", "sym5").
    max_level : int
        Decomposition level.
    n_bands : int
        Max number of leaf nodes to return (top-energy nodes first).

    Returns
    -------
    matrix : ndarray (n_bands, n_samples)
        Reconstructed signals from each WPT leaf node.
    freq_axes : ndarray (n_bands,)
        Nominal center frequencies of each node band.
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="symmetric",
                            maxlevel=max_level)
    nodes = wp.get_level(max_level, "natural")
    leaf_names = [node.path for node in nodes]
    leaf_signals = np.array([wp[node_name].data for node_name in leaf_names])  # (n_leaves, n_len)

    # Each leaf covers a frequency sub-band: fs/2 split per level -> fs/2*(node_idx/(2**level))
    n_leaves = len(nodes)
    nyq = fs / 2.0
    bin_width = nyq / (2 ** max_level)

    # Select top-energy nodes
    node_energy = np.mean(leaf_signals**2, axis=1)
    top_k = min(n_bands, n_leaves)
    top_idx = np.argsort(node_energy)[-top_k:]
    top_idx = np.sort(top_idx)

    matrix = leaf_signals[top_idx]

    # Calculate frequency axes from node paths
    # PyWavelets uses 'a' and 'd' for approximation and detail coefficients
    freq_axes = []
    for i in top_idx:
        path = leaf_names[i]
        # Convert path to binary index: 'a' -> 0, 'd' -> 1
        binary_str = path.replace('a', '0').replace('d', '1')
        node_idx = int(binary_str, 2)
        freq_axes.append((node_idx + 0.5) * bin_width)
    freq_axes = np.array(freq_axes)

    return matrix, freq_axes
