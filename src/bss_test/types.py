"""
类型定义模块
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# 信号类型
Signal1D = np.ndarray  # 1D 信号 (n_samples,)
Signal2D = np.ndarray  # 2D 信号 (n_channels, n_samples)
Frequency = float      # 频率值
SampleRate = float     # 采样率
RPM = int              # 转速

# 配置类型
ConfigDict = Dict[str, Any]
FaultFreqs = Dict[str, float]

# 结果类型
BSSResult = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (S_est, A_est, W)
TFAResult = Tuple[np.ndarray, np.ndarray]              # (matrix, freqs)
CWTResult = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (coefficients, frequencies, scales)
