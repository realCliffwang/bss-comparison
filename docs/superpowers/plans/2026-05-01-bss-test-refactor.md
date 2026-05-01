# BSS-Test 项目重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 BSS-Test 项目结构，提升可维护性、开发者体验、扩展性和代码质量

**Architecture:** 创建 `src/bss_test/` 包结构，将核心模块拆分为子包（io、tfa、bss、utils），辅助模块保持单文件，统一配置管理和导入方式，实验脚本重组到 `experiments/` 目录

**Tech Stack:** Python 3.8+、dataclasses、PyYAML、numpy、scipy

---

## Task 1: 创建包结构和基础文件

**Files:**
- Create: `src/bss_test/__init__.py`
- Create: `src/bss_test/types.py`
- Create: `src/bss_test/io/__init__.py`
- Create: `src/bss_test/tfa/__init__.py`
- Create: `src/bss_test/bss/__init__.py`
- Create: `src/bss_test/utils/__init__.py`

- [ ] **Step 1: 创建 bss_test 主包目录**

```bash
mkdir -p src/bss_test/io
mkdir -p src/bss_test/tfa
mkdir -p src/bss_test/bss
mkdir -p src/bss_test/utils
```

- [ ] **Step 2: 创建主包 __init__.py**

```python
# src/bss_test/__init__.py
"""
BSS-Test: 轴承故障诊断盲源分离框架
"""

__version__ = "0.3.0"
__author__ = "BSS-Test Contributors"

# 常用导入快捷方式
from bss_test.preprocessing import preprocess_signals
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
)
from bss_test.utils.config import ExperimentConfig
from bss_test.utils.logger import get_logger, setup_logging
from bss_test.utils.exceptions import (
    BSSTestError,
    DataLoadError,
    PreprocessingError,
    TFAError,
    BSSError,
    FeatureExtractionError,
    ClassifierError,
    ConfigurationError,
)

# 版本信息
VERSION_INFO = {
    "version": __version__,
    "major": 0,
    "minor": 3,
    "patch": 0,
}


def get_version() -> str:
    """获取当前版本字符串。"""
    return __version__


def get_version_info() -> dict:
    """获取详细版本信息。"""
    return VERSION_INFO.copy()
```

- [ ] **Step 3: 创建 types.py 类型定义**

```python
# src/bss_test/types.py
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
```

- [ ] **Step 4: 创建子包 __init__.py**

```python
# src/bss_test/io/__init__.py
"""
数据 I/O 子包
"""

from bss_test.io.cwru import load_cwru, list_cwru_files
from bss_test.io.phm import load_phm_cut, load_phm_wear
from bss_test.io.nasa import load_nasa_milling_single

__all__ = [
    "load_cwru",
    "list_cwru_files",
    "load_phm_cut",
    "load_phm_wear",
    "load_nasa_milling_single",
]
```

```python
# src/bss_test/tfa/__init__.py
"""
时频分析子包
"""

from bss_test.tfa.factory import time_freq_factory, build_observation_matrix
from bss_test.tfa.cwt import cwt_transform

__all__ = [
    "time_freq_factory",
    "build_observation_matrix",
    "cwt_transform",
]
```

```python
# src/bss_test/bss/__init__.py
"""
盲源分离子包
"""

from bss_test.bss.factory import bss_factory

__all__ = ["bss_factory"]
```

```python
# src/bss_test/utils/__init__.py
"""
工具子包
"""

from bss_test.utils.config import ExperimentConfig
from bss_test.utils.logger import get_logger, setup_logging
from bss_test.utils.exceptions import BSSTestError

__all__ = [
    "ExperimentConfig",
    "get_logger",
    "setup_logging",
    "BSSTestError",
]
```

- [ ] **Step 5: Commit**

```bash
git add src/bss_test/
git commit -m "feat: 创建 bss_test 包结构和基础文件"
```

---

## Task 2: 迁移工具模块到 utils/

**Files:**
- Create: `src/bss_test/utils/config.py`
- Create: `src/bss_test/utils/logger.py`
- Create: `src/bss_test/utils/exceptions.py`
- Delete: `src/config.py`
- Delete: `src/logger.py`
- Delete: `src/exceptions.py`

- [ ] **Step 1: 复制 config.py 到 utils/ 并更新导入**

```bash
cp src/config.py src/bss_test/utils/config.py
```

更新文件中的导入路径：
- 将 `from src.` 改为 `from bss_test.`

- [ ] **Step 2: 复制 logger.py 到 utils/ 并更新导入**

```bash
cp src/logger.py src/bss_test/utils/logger.py
```

- [ ] **Step 3: 复制 exceptions.py 到 utils/ 并更新导入**

```bash
cp src/exceptions.py src/bss_test/utils/exceptions.py
```

- [ ] **Step 4: 删除旧文件**

```bash
rm src/config.py
rm src/logger.py
rm src/exceptions.py
```

- [ ] **Step 5: Commit**

```bash
git add src/bss_test/utils/ src/config.py src/logger.py src/exceptions.py
git commit -m "refactor: 迁移工具模块到 utils/"
```

---

## Task 3: 迁移数据加载模块到 io/

**Files:**
- Create: `src/bss_test/io/cwru.py`
- Create: `src/bss_test/io/phm.py`
- Create: `src/bss_test/io/nasa.py`
- Delete: `src/data_loader.py`

- [ ] **Step 1: 分析 data_loader.py 结构**

读取 `src/data_loader.py`，识别：
- CWRU 相关函数
- PHM 相关函数
- NASA 相关函数

- [ ] **Step 2: 创建 cwru.py**

提取 CWRU 相关代码：
- `CWRU_FAULT_MAP`
- `list_cwru_files()`
- `load_cwru()`

```python
# src/bss_test/io/cwru.py
"""
CWRU 轴承数据加载器
"""

import os
import numpy as np
from scipy.io import loadmat
from typing import List, Dict, Optional, Tuple

# CWRU 故障映射表
CWRU_FAULT_MAP = {
    ("normal", 0): "109",
    ("inner_race_007", 0): "122",
    ("ball_007", 0): "135",
    ("outer_race_6_007", 0): "148",
    # ... 其他映射
}


def list_cwru_files(data_dir: str = "data/cwru") -> List[Dict]:
    """扫描数据目录并列出可用的 .mat 文件"""
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    files.sort()
    result = []
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            mat = loadmat(path)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            result.append({"file": f, "keys": keys, "path": path})
        except Exception as e:
            result.append({"file": f, "error": str(e), "path": path})
    return result


def load_cwru(
    data_dir: str = "data/cwru",
    fault_type: str = "inner_race_007",
    load: int = 0,
    channels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    加载 CWRU 轴承振动数据
    
    Parameters
    ----------
    data_dir : str
        包含 .mat 文件的目录路径
    fault_type : str
        故障类型
    load : int
        负载等级 (0-3)
    channels : list[str] or None
        通道列表
        
    Returns
    -------
    signals : np.ndarray
        信号数据 (n_channels, n_samples)
    fs : float
        采样率
    rpm : int
        转速
    """
    # ... 实现代码
```

- [ ] **Step 3: 创建 phm.py**

提取 PHM 相关代码：
- `load_phm_cut()`
- `load_phm_wear()`

- [ ] **Step 4: 创建 nasa.py**

提取 NASA 相关代码：
- `load_nasa_milling()`
- `load_nasa_milling_single()`

- [ ] **Step 5: 删除旧文件**

```bash
rm src/data_loader.py
```

- [ ] **Step 6: Commit**

```bash
git add src/bss_test/io/ src/data_loader.py
git commit -m "refactor: 迁移数据加载模块到 io/"
```

---

## Task 4: 迁移时频分析模块到 tfa/

**Files:**
- Create: `src/bss_test/tfa/factory.py`
- Create: `src/bss_test/tfa/cwt.py`
- Create: `src/bss_test/tfa/stft.py`
- Create: `src/bss_test/tfa/wpt.py`
- Create: `src/bss_test/tfa/emd.py`
- Delete: `src/cwt_module.py`

- [ ] **Step 1: 分析 cwt_module.py 结构**

读取 `src/cwt_module.py`，识别：
- CWT 相关函数
- STFT 相关函数
- WPT 相关函数
- EMD 相关函数
- 工厂函数
- 绘图函数

- [ ] **Step 2: 创建 cwt.py**

提取 CWT 相关代码：
- `cwt_transform()`
- `plot_cwt_spectrogram()`

```python
# src/bss_test/tfa/cwt.py
"""
连续小波变换 (CWT) 模块
"""

import numpy as np
import pywt
from typing import Tuple, Optional


def cwt_transform(
    signal: np.ndarray,
    fs: float,
    wavelet: str = "cmor1.5-1.0",
    n_bands: int = 20,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    连续小波变换
    
    Parameters
    ----------
    signal : np.ndarray
        输入信号 (n_samples,)
    fs : float
        采样率
    wavelet : str
        小波名称
    n_bands : int
        频带数量
    freq_range : tuple or None
        频率范围 (low, high)
        
    Returns
    -------
    coefficients : np.ndarray
        小波系数 (n_bands, n_samples)
    frequencies : np.ndarray
        频率值 (n_bands,)
    scales : np.ndarray
        尺度值 (n_bands,)
    """
    # ... 实现代码
```

- [ ] **Step 3: 创建 stft.py**

提取 STFT 相关代码：
- `stft_transform()`

- [ ] **Step 4: 创建 wpt.py**

提取 WPT 相关代码：
- `wpt_transform()`

- [ ] **Step 5: 创建 emd.py**

提取 EMD 相关代码：
- `emd_transform()`
- `eemd_transform()`
- `ceemdan_transform()`

- [ ] **Step 6: 创建 factory.py**

提取工厂函数：
- `time_freq_factory()`
- `build_observation_matrix()`

```python
# src/bss_test/tfa/factory.py
"""
时频分析工厂模块
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def time_freq_factory(
    signal: np.ndarray,
    fs: float,
    method: str = "cwt",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    时频分析工厂函数
    
    Parameters
    ----------
    signal : np.ndarray
        输入信号
    fs : float
        采样率
    method : str
        分析方法 (cwt, stft, wpt, emd, eemd, ceemdan)
    **kwargs : dict
        方法特定参数
        
    Returns
    -------
    matrix : np.ndarray
        时频矩阵
    freqs : np.ndarray
        频率值
    """
    if method == "cwt":
        from bss_test.tfa.cwt import cwt_transform
        coef, freqs, scales = cwt_transform(signal, fs, **kwargs)
        return coef, freqs
    elif method == "stft":
        from bss_test.tfa.stft import stft_transform
        return stft_transform(signal, fs, **kwargs)
    elif method == "wpt":
        from bss_test.tfa.wpt import wpt_transform
        return wpt_transform(signal, fs, **kwargs)
    elif method in ["emd", "eemd", "ceemdan"]:
        from bss_test.tfa.emd import emd_transform
        return emd_transform(signal, fs, method=method, **kwargs)
    else:
        raise ValueError(f"未知的时频分析方法: {method}")


def build_observation_matrix(
    signals: np.ndarray,
    fs: float,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, list]:
    """
    构建观测矩阵
    
    Parameters
    ----------
    signals : np.ndarray
        输入信号 (n_channels, n_samples)
    fs : float
        采样率
    config : dict
        配置字典
        
    Returns
    -------
    matrix : np.ndarray
        观测矩阵 (n_obs, n_samples)
    labels : list
        观测标签
    """
    # ... 实现代码
```

- [ ] **Step 7: 删除旧文件**

```bash
rm src/cwt_module.py
```

- [ ] **Step 8: Commit**

```bash
git add src/bss_test/tfa/ src/cwt_module.py
git commit -m "refactor: 迁移时频分析模块到 tfa/"
```

---

## Task 5: 迁移盲源分离模块到 bss/

**Files:**
- Create: `src/bss_test/bss/factory.py`
- Create: `src/bss_test/bss/sobi.py`
- Create: `src/bss_test/bss/ica.py`
- Create: `src/bss_test/bss/jade.py`
- Delete: `src/bss_module.py`

- [ ] **Step 1: 分析 bss_module.py 结构**

读取 `src/bss_module.py`，识别：
- SOBI 相关函数
- ICA 相关函数
- JADE 相关函数
- 工厂函数

- [ ] **Step 2: 创建 sobi.py**

提取 SOBI 相关代码：
- `run_sobi()`

```python
# src/bss_test/bss/sobi.py
"""
SOBI 盲源分离算法
"""

import numpy as np
from typing import Tuple


def run_sobi(
    X: np.ndarray,
    n_sources: int = 5,
    n_lags: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    二阶盲辨识 (SOBI) 算法
    
    Parameters
    ----------
    X : np.ndarray
        观测矩阵 (n_obs, n_samples)
    n_sources : int
        源数量
    n_lags : int
        时滞数量
        
    Returns
    -------
    S_est : np.ndarray
        估计源信号 (n_sources, n_samples)
    W : np.ndarray
        分离矩阵 (n_sources, n_obs)
    """
    # ... 实现代码
```

- [ ] **Step 3: 创建 ica.py**

提取 ICA 相关代码：
- `run_fastica()`
- `run_picard()`

- [ ] **Step 4: 创建 jade.py**

提取 JADE 相关代码：
- `run_jade()`

- [ ] **Step 5: 创建 factory.py**

提取工厂函数：
- `bss_factory()`

```python
# src/bss_test/bss/factory.py
"""
盲源分离工厂模块
"""

import numpy as np
from typing import Tuple, Optional


def bss_factory(
    X: np.ndarray,
    method: str = "sobi",
    n_components: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    盲源分离工厂函数
    
    Parameters
    ----------
    X : np.ndarray
        观测矩阵 (n_obs, n_samples)
    method : str
        BSS 方法 (sobi, fastica, jade, picard, nmf, pca)
    n_components : int or None
        源数量
    **kwargs : dict
        方法特定参数
        
    Returns
    -------
    S_est : np.ndarray
        估计源信号 (n_components, n_samples)
    A_est : np.ndarray
        估计混合矩阵 (n_obs, n_components)
    W : np.ndarray
        分离矩阵 (n_components, n_obs)
    """
    if n_components is None:
        n_components = X.shape[0]
    
    if method == "sobi":
        from bss_test.bss.sobi import run_sobi
        S_est, W = run_sobi(X, n_sources=n_components, **kwargs)
        A_est = np.linalg.pinv(W)
        return S_est, A_est, W
    elif method == "fastica":
        from bss_test.bss.ica import run_fastica
        return run_fastica(X, n_components=n_components, **kwargs)
    elif method == "jade":
        from bss_test.bss.jade import run_jade
        return run_jade(X, n_components=n_components, **kwargs)
    elif method == "picard":
        from bss_test.bss.ica import run_picard
        return run_picard(X, n_components=n_components, **kwargs)
    else:
        raise ValueError(f"未知的 BSS 方法: {method}")
```

- [ ] **Step 6: 删除旧文件**

```bash
rm src/bss_module.py
```

- [ ] **Step 7: Commit**

```bash
git add src/bss_test/bss/ src/bss_module.py
git commit -m "refactor: 迁移盲源分离模块到 bss/"
```

---

## Task 6: 保留单文件模块

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `src/feature_extractor.py`
- Modify: `src/ml_classifier.py`
- Modify: `src/evaluation.py`

- [ ] **Step 1: 更新 preprocessing.py 导入**

更新文件中的导入路径：
- 将 `from src.` 改为 `from bss_test.`

```python
# src/preprocessing.py
"""
信号预处理模块
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Optional, Tuple

# 更新导入
from bss_test.utils.logger import get_logger

logger = get_logger(__name__)

# ... 其余代码保持不变
```

- [ ] **Step 2: 更新 feature_extractor.py 导入**

- [ ] **Step 3: 更新 ml_classifier.py 导入**

- [ ] **Step 4: 更新 evaluation.py 导入**

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py src/feature_extractor.py src/ml_classifier.py src/evaluation.py
git commit -m "refactor: 更新单文件模块导入路径"
```

---

## Task 7: 创建 YAML 配置文件

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/cwru.yaml`
- Create: `configs/phm2010.yaml`
- Create: `configs/nasa.yaml`

- [ ] **Step 1: 创建 configs 目录**

```bash
mkdir -p configs
```

- [ ] **Step 2: 创建 default.yaml**

```yaml
# configs/default.yaml
name: default_experiment
data_dir: data
output_dir: outputs

preprocess:
  detrend: true
  bandpass: [100, 5000]
  normalize: zscore
  resample_fs: null

tfa:
  method: cwt
  wavelet: cmor1.5-1.0
  n_bands: 20
  freq_range: [100, 5000]
  mode: single_channel_expansion

bss:
  method: SOBI
  n_sources: 5
  n_lags: 50

feature_freqs:
  BPFO: null
  BPFI: null
  BSF: null
  FTF: null
```

- [ ] **Step 3: 创建 cwru.yaml**

```yaml
# configs/cwru.yaml
name: cwru_experiment
data_dir: data/cwru
output_dir: outputs/cwru

preprocess:
  detrend: true
  bandpass: [100, 5000]
  normalize: zscore

tfa:
  method: cwt
  wavelet: cmor1.5-1.0
  n_bands: 20
  freq_range: [100, 5000]
  mode: single_channel_expansion

bss:
  method: SOBI
  n_sources: 5
  n_lags: 50

feature_freqs:
  BPFO: 107.3
  BPFI: 162.2
  BSF: 70.6
```

- [ ] **Step 4: 创建 phm2010.yaml**

```yaml
# configs/phm2010.yaml
name: phm2010_experiment
data_dir: data/phm2010_milling
output_dir: outputs/phm2010

preprocess:
  detrend: true
  bandpass: [100, 20000]
  normalize: zscore

tfa:
  method: cwt
  wavelet: cmor1.5-1.0
  n_bands: 20
  freq_range: [100, 20000]
  mode: multi_channel
  bands_per_ch: 7

bss:
  method: SOBI
  n_sources: 6
  n_lags: 50

feature_freqs:
  spindle: 173.3
```

- [ ] **Step 5: 创建 nasa.yaml**

```yaml
# configs/nasa.yaml
name: nasa_experiment
data_dir: data/phm2010_milling
output_dir: outputs/nasa_milling

preprocess:
  detrend: true
  bandpass: [10, 100]
  normalize: zscore

tfa:
  method: cwt
  wavelet: cmor1.5-1.0
  n_bands: 20
  freq_range: [5, 100]
  mode: multi_channel

bss:
  method: SOBI
  n_sources: 5
  n_lags: 50

feature_freqs:
  spindle: 166.7
```

- [ ] **Step 6: Commit**

```bash
git add configs/
git commit -m "feat: 创建 YAML 配置文件"
```

---

## Task 8: 重组实验脚本

**Files:**
- Create: `experiments/single/cwru.py`
- Create: `experiments/single/phm_milling.py`
- Create: `experiments/single/nasa_milling.py`
- Create: `experiments/comparison/bss_methods.py`
- Create: `experiments/comparison/tfa_methods.py`
- Create: `experiments/comparison/full.py`
- Create: `experiments/reports/comparison_report.py`
- Create: `experiments/reports/summary.py`

- [ ] **Step 1: 创建 experiments 目录结构**

```bash
mkdir -p experiments/single
mkdir -p experiments/comparison
mkdir -p experiments/reports
```

- [ ] **Step 2: 创建 experiments/single/cwru.py**

```python
# experiments/single/cwru.py
"""
CWRU 轴承故障诊断实验
用法: python -m experiments.single.cwru
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test import (
    load_cwru,
    preprocess_signals,
    ExperimentConfig,
)
from bss_test.tfa import build_observation_matrix, cwt_transform
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_correlation_matrix,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """运行 CWRU 实验"""
    # 加载配置
    config = ExperimentConfig.from_yaml("configs/cwru.yaml")
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("CWRU 轴承故障诊断实验")
    logger.info("=" * 60)
    
    # 加载数据
    logger.info("\n[1/5] 加载 CWRU 数据...")
    signals, fs, rpm = load_cwru(
        data_dir=config.data_dir,
        fault_type="inner_race_007",
        load=0,
        channels=["DE"],
    )
    logger.info(f"  加载 {signals.shape[0]} 通道, {signals.shape[1]} 样本 @ {fs} Hz")
    
    # 使用子集进行快速处理
    n_use = min(signals.shape[1], int(2.0 * fs))
    signals = signals[:, :n_use]
    logger.info(f"  使用前 {n_use} 样本 ({n_use/fs:.2f} 秒)")
    
    # 预处理
    logger.info("\n[2/5] 预处理...")
    signals_pre, fs_pre = preprocess_signals(signals, fs, config.preprocess)
    logger.info(f"  预处理后形状: {signals_pre.shape}, fs: {fs_pre} Hz")
    
    # 构建观测矩阵
    logger.info("\n[3/5] 构建观测矩阵...")
    X, labels = build_observation_matrix(signals_pre, fs_pre, config.tfa)
    logger.info(f"  观测矩阵: {X.shape[0]} 观测 × {X.shape[1]} 样本")
    
    # BSS 分离
    logger.info(f"\n[4/5] 运行 BSS ({config.bss.method})...")
    S_est, A_est, W = bss_factory(
        X, 
        method=config.bss.method, 
        n_components=config.bss.n_sources
    )
    logger.info(f"  估计源: {S_est.shape[0]} × {S_est.shape[1]}")
    
    # 评估
    logger.info("\n[5/5] 评估...")
    indep = compute_independence_metric(S_est)
    ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)
    
    logger.info(f"  独立性指标: {indep:.4f}")
    logger.info(f"  故障频率检测分数: {ffds:.2f}")
    
    # 可视化
    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre,
        fault_freqs=config.feature_freqs,
        title_prefix=f"CWRU inner_race_007 — "
    )
    fig.savefig(output_dir / "envelope_spectrum.png", dpi=150)
    plt.close(fig)
    
    fig, _ = plot_correlation_matrix(S_est, title="CWRU Source Correlation")
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    
    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
```

- [ ] **Step 3: 创建其他实验脚本**

类似地创建：
- `experiments/single/phm_milling.py`
- `experiments/single/nasa_milling.py`
- `experiments/comparison/bss_methods.py`
- `experiments/comparison/tfa_methods.py`
- `experiments/comparison/full.py`
- `experiments/reports/comparison_report.py`
- `experiments/reports/summary.py`

- [ ] **Step 4: Commit**

```bash
git add experiments/
git commit -m "feat: 重组实验脚本到 experiments/"
```

---

## Task 9: 更新打包配置和清理

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `setup.py`
- Delete: `main_cwru.py`
- Delete: `main_phm_milling.py`
- Delete: `main_nasa_milling.py`
- Delete: `run_bss_comparison.py`
- Delete: `run_full_comparison.py`
- Delete: `run_full_comparison_v2.py`
- Delete: `run_tfa_comparison.py`
- Delete: `generate_comparison_report.py`
- Delete: `generate_summary.py`

- [ ] **Step 1: 更新 pyproject.toml**

更新包路径配置：

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "bss-test"
version = "0.3.0"
# ... 其他配置保持不变

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "-p", "no:warnings",
]

[tool.coverage.run]
source = ["src/bss_test"]
omit = [
    "tests/*",
    "src/bss_test/tfa/emd.py",
]
```

- [ ] **Step 2: 更新 Makefile**

更新实验运行命令：

```makefile
# Running experiments
run-cwru:
	python experiments/single/cwru.py

run-phm:
	python experiments/single/phm_milling.py

run-nasa:
	python experiments/single/nasa_milling.py

run-comparison:
	python experiments/comparison/bss_methods.py

run-summary:
	python experiments/reports/summary.py

run-all: run-cwru run-phm run-nasa run-comparison
```

- [ ] **Step 3: 删除旧脚本**

```bash
rm main_cwru.py
rm main_phm_milling.py
rm main_nasa_milling.py
rm run_bss_comparison.py
rm run_full_comparison.py
rm run_full_comparison_v2.py
rm run_tfa_comparison.py
rm generate_comparison_report.py
rm generate_summary.py
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml Makefile setup.py
git rm main_cwru.py main_phm_milling.py main_nasa_milling.py
git rm run_bss_comparison.py run_full_comparison.py run_full_comparison_v2.py
git rm run_tfa_comparison.py generate_comparison_report.py generate_summary.py
git commit -m "refactor: 更新打包配置和清理旧脚本"
```

---

## Task 10: 最终验证

**Files:**
- Verify: 所有模块导入正常
- Verify: 配置加载正常
- Verify: 实验脚本可运行

- [ ] **Step 1: 测试包导入**

```bash
python -c "from bss_test import get_version; print(get_version())"
```

Expected: `0.3.0`

- [ ] **Step 2: 测试配置加载**

```bash
python -c "from bss_test.utils.config import ExperimentConfig; config = ExperimentConfig.from_yaml('configs/cwru.yaml'); print(config.name)"
```

Expected: `cwru_experiment`

- [ ] **Step 3: 测试实验脚本**

```bash
python experiments/single/cwru.py
```

Expected: 正常运行并输出结果

- [ ] **Step 4: 运行测试**

```bash
pytest tests/ -v
```

Expected: 所有测试通过

- [ ] **Step 5: 最终 Commit**

```bash
git add -A
git commit -m "refactor: 完成 BSS-Test 项目重构"
```

---

## 执行选项

**计划完成并保存到 `docs/superpowers/plans/2026-05-01-bss-test-refactor.md`。两种执行选项：**

**1. Subagent-Driven (推荐)** - 每个任务分发给独立的 subagent 执行，任务间进行审查，快速迭代

**2. Inline Execution** - 在当前会话中执行任务，批量执行并设置检查点

**选择哪种方式？**
