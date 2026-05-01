"""
时频分析工厂模块
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


def time_freq_factory(
    signal: np.ndarray,
    fs: float,
    method: str = "cwt",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一时频分析工厂函数

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    method : str
        分析方法 (cwt, stft, wpt, vmd, emd, eemd, ceemdan)
    **kwargs : dict
        方法特定参数

    Returns
    -------
    matrix : ndarray (n_bands, n_samples)
        时频矩阵
    freqs : ndarray (n_bands,)
        频率值
    """
    method = method.lower()

    if method == "cwt":
        from bss_test.tfa.cwt import cwt_transform
        n_bands = kwargs.pop("n_bands", 20)
        freq_range = kwargs.pop("freq_range", None)
        wavelet = kwargs.pop("wavelet", "cmor1.5-1.0")
        coef, freqs, _scales = cwt_transform(
            signal, fs, wavelet=wavelet, n_bands=n_bands, freq_range=freq_range
        )
        return coef, freqs

    elif method == "stft":
        from bss_test.tfa.stft import stft_transform
        kwargs.pop("wavelet", None)
        kwargs.pop("bands_per_ch", None)
        return stft_transform(signal, fs, **kwargs)

    elif method == "wpt":
        from bss_test.tfa.wpt import wpt_transform
        kwargs.pop("bands_per_ch", None)
        return wpt_transform(signal, fs, **kwargs)

    elif method == "vmd":
        from bss_test.tfa.emd import vmd_transform
        kwargs.pop("wavelet", None)
        kwargs.pop("freq_range", None)
        kwargs.pop("bands_per_ch", None)
        return vmd_transform(signal, fs, **kwargs)

    elif method in ["emd", "eemd", "ceemdan"]:
        from bss_test.tfa.emd import emd_factory
        kwargs.pop("wavelet", None)
        kwargs.pop("freq_range", None)
        kwargs.pop("bands_per_ch", None)
        return emd_factory(signal, fs, method=method, **kwargs)

    else:
        raise ValueError(
            f"未知的 TFA 方法: {method}. "
            f"支持的方法: cwt, stft, wpt, vmd, emd, eemd, ceemdan"
        )


def build_observation_matrix(
    signals_pre: np.ndarray,
    fs: float,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    构建观测矩阵 X_for_bss via TFA + 频带选择。

    两种模式:
    1. "multi_channel": 每个物理通道 -> TFA -> 选择 K 个频带 ->
                        拼接 (n_ch * K, n_samples).
    2. "single_channel_expansion": 单通道 -> TFA -> 选择 K 个频带 ->
                        每个频带作为伪通道 -> (K, n_samples).

    Parameters
    ----------
    signals_pre : ndarray (n_channels, n_samples)
        预处理后的信号.
    fs : float
        采样率.
    config : dict or None
        配置字典，包含以下键:
        - mode : "single_channel_expansion" 或 "multi_channel" (必需)
        - tfa_method : "cwt" (默认) | "stft" | "wpt" | "vmd" | "emd" |
                       "eemd" | "ceemdan"
        - n_bands : int (默认 20)
        - freq_range : tuple (low, high) 或 None
        - wavelet : str (默认 "cmor1.5-1.0") — 用于 CWT 和 WPT
        - band_indices : list of int 或 None — 显式频带选择
        - bands_per_ch : int — 多通道模式下每个通道的频带数
        - return_complex : bool — 如果为 True，使用复数系数 (仅 CWT)

    Returns
    -------
    X_for_bss : ndarray (n_obs, n_samples)
        用于 BSS 的观测矩阵.
    obs_labels : list of str
        描述每个观测的标签 (例如 "ch0_f123Hz").
    """
    if config is None:
        config = {}

    mode = config.get("mode", "single_channel_expansion")
    n_bands = config.get("n_bands", 20)
    freq_range = config.get("freq_range", None)
    wavelet = config.get("wavelet", "cmor1.5-1.0")
    band_indices = config.get("band_indices", None)
    tfa_method = config.get("tfa_method", "cwt")

    if mode == "single_channel_expansion":
        signal_1d = signals_pre[0]  # 使用第一个通道
        coef, freqs = time_freq_factory(
            signal_1d, fs, method=tfa_method,
            n_bands=n_bands, freq_range=freq_range, wavelet=wavelet
        )

        if band_indices is not None:
            coef = coef[band_indices]
        else:
            band_indices = list(range(coef.shape[0]))

        X_for_bss = coef  # shape (K, n_samples)
        obs_labels = [f"ch0_f{freqs[i]:.0f}Hz" for i in band_indices]

    elif mode == "multi_channel":
        n_channels = signals_pre.shape[0]
        bands_per_ch = config.get("bands_per_ch", n_bands // n_channels)
        blocks = []
        obs_labels = []

        for ch in range(n_channels):
            coef, freqs = time_freq_factory(
                signals_pre[ch], fs, method=tfa_method,
                n_bands=bands_per_ch, freq_range=freq_range, wavelet=wavelet
            )
            if band_indices is not None and ch == 0:
                ch_indices = band_indices
            else:
                ch_indices = list(range(coef.shape[0]))
            blocks.append(coef[ch_indices])
            obs_labels += [f"ch{ch}_f{freqs[i]:.0f}Hz" for i in ch_indices]

        X_for_bss = np.vstack(blocks)

    else:
        raise ValueError(f"未知模式: {mode}. 使用 'single_channel_expansion' 或 'multi_channel'.")

    return X_for_bss, obs_labels
