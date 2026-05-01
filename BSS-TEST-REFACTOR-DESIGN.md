# BSS-Test 项目重构设计文档

> **日期:** 2026-05-01
> **版本:** 1.0
> **状态:** 已批准

---

## 1. 概述

### 1.1 重构目标

1. **提升可维护性** - 让代码更容易理解和修改
2. **改善开发者体验** - 统一的导入方式、配置管理
3. **为未来扩展做准备** - 更容易添加新数据集、新算法、新功能
4. **提升代码质量** - 更好的模块组织和职责划分

### 1.2 约束条件

- 向后兼容性：不需要兼容，可自由重构
- 打包分发：暂时不需要正式打包
- 测试：暂时不增加测试，重构完成后再补充

---

## 2. 目录结构

```
BSS-test/
├── pyproject.toml                  # 打包配置（保留）
├── Makefile                        # 构建命令（保留）
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/                        # YAML 配置文件
│   ├── default.yaml               # 默认配置
│   ├── cwru.yaml                  # CWRU 数据集配置
│   ├── phm2010.yaml               # PHM 2010 配置
│   └── nasa.yaml                  # NASA 铣削配置
│
├── src/
│   └── bss_test/                   # 主包
│       ├── __init__.py
│       ├── types.py                # 类型定义
│       │
│       ├── io/                     # 数据 I/O 子包
│       │   ├── __init__.py
│       │   ├── cwru.py             # CWRU 数据加载
│       │   ├── phm.py              # PHM 2010 数据加载
│       │   └── nasa.py             # NASA 数据加载
│       │
│       ├── tfa/                    # 时频分析子包
│       │   ├── __init__.py
│       │   ├── factory.py          # TFA 工厂函数
│       │   ├── cwt.py              # CWT 实现
│       │   ├── stft.py             # STFT 实现
│       │   ├── wpt.py              # WPT 实现
│       │   └── emd.py              # EMD/EEMD/CEEMDAN 实现
│       │
│       ├── bss/                    # 盲源分离子包
│       │   ├── __init__.py
│       │   ├── factory.py          # BSS 工厂函数
│       │   ├── sobi.py             # SOBI 实现
│       │   ├── ica.py              # FastICA/PICARD 实现
│       │   └── jade.py             # JADE 实现
│       │
│       ├── preprocessing.py        # 信号预处理（保持单文件）
│       ├── feature_extractor.py    # 特征提取（保持单文件）
│       ├── ml_classifier.py        # ML 分类器（保持单文件）
│       ├── evaluation.py           # 评估指标 + 可视化（保持单文件）
│       │
│       └── utils/                  # 工具子包
│           ├── __init__.py
│           ├── config.py           # 配置管理
│           ├── logger.py           # 日志工具
│           └── exceptions.py       # 自定义异常
│
├── experiments/                    # 实验脚本
│   ├── single/                     # 单数据集实验
│   │   ├── cwru.py
│   │   ├── phm_milling.py
│   │   └── nasa_milling.py
│   │
│   ├── comparison/                 # 对比实验
│   │   ├── bss_methods.py
│   │   ├── tfa_methods.py
│   │   └── full.py
│   │
│   └── reports/                    # 报告生成
│       ├── comparison_report.py
│       └── summary.py
│
├── tests/                          # 测试（暂时保持现状）
├── data/                           # 数据（gitignore）
└── outputs/                        # 输出（gitignore）
```

---

## 3. 模块职责与接口

### 3.1 数据 I/O 模块 (`io/`)

**职责：** 只负责数据加载和解析，不做预处理。

```python
# io/cwru.py
def load_cwru(
    data_dir: str = "data/cwru",
    fault_type: str = "inner_race_007",
    load: int = 0,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, float, int]:
    """返回 (signals, fs, rpm)"""

def list_cwru_files(data_dir: str = "data/cwru") -> list[dict]:
    """列出可用的 CWRU 文件"""

# io/phm.py
def load_phm_cut(
    tool_id: int,
    cut_no: int,
    data_dir: str = "data/phm2010_milling",
    sensor_types: list[str] | None = None,
) -> tuple[np.ndarray, float, dict]:
    """返回 (signals, fs, metadata)"""

def load_phm_wear(tool_id: int, data_dir: str = "data/phm2010_milling") -> np.ndarray:
    """返回磨损数据"""

# io/nasa.py
def load_nasa_milling_single(
    run_index: int,
    data_dir: str = "data/phm2010_milling",
    sensor_types: list[str] | None = None,
) -> tuple[np.ndarray, dict, float]:
    """返回 (signals, metadata, fs)"""
```

### 3.2 时频分析模块 (`tfa/`)

**职责：** 时频变换和观测矩阵构建，不做 BSS 分离。

```python
# tfa/factory.py
def time_freq_factory(
    signal: np.ndarray,
    fs: float,
    method: str = "cwt",  # cwt, stft, wpt, emd, eemd, ceemdan
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """统一入口，返回 (matrix, freqs)"""

def build_observation_matrix(
    signals: np.ndarray,
    fs: float,
    config: dict,
) -> tuple[np.ndarray, list[str]]:
    """构建观测矩阵，返回 (matrix, labels)"""

# tfa/cwt.py
def cwt_transform(
    signal: np.ndarray,
    fs: float,
    wavelet: str = "cmor1.5-1.0",
    n_bands: int = 20,
    freq_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (coefficients, frequencies, scales)"""
```

### 3.3 盲源分离模块 (`bss/`)

**职责：** 只负责源分离，不做特征提取或评估。

```python
# bss/factory.py
def bss_factory(
    X: np.ndarray,
    method: str = "sobi",  # sobi, fastica, jade, picard, nmf, pca
    n_components: int | None = None,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """统一入口，返回 (S_est, A_est, W)"""

# bss/sobi.py
def run_sobi(
    X: np.ndarray,
    n_sources: int = 5,
    n_lags: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (S_est, W)"""

# bss/ica.py
def run_fastica(X: np.ndarray, n_components: int | None = None, **kwargs) -> tuple:
def run_picard(X: np.ndarray, n_components: int | None = None, **kwargs) -> tuple:

# bss/jade.py
def run_jade(X: np.ndarray, n_components: int | None = None) -> tuple:
```

### 3.4 辅助模块（保持单文件）

- **`preprocessing.py`** - 信号预处理
- **`feature_extractor.py`** - 特征提取
- **`ml_classifier.py`** - ML 分类器
- **`evaluation.py`** - 评估指标 + 可视化

---

## 4. 配置管理系统

### 4.1 配置结构（dataclass）

```python
# src/bss_test/utils/config.py

@dataclass
class PreprocessConfig:
    detrend: bool = True
    bandpass: Optional[tuple[float, float]] = (100, 5000)
    normalize: str = "zscore"
    resample_fs: Optional[float] = None

@dataclass
class TFAConfig:
    method: str = "cwt"
    wavelet: str = "cmor1.5-1.0"
    n_bands: int = 20
    freq_range: Optional[tuple[float, float]] = (100, 5000)
    mode: str = "single_channel_expansion"

@dataclass
class BSSConfig:
    method: str = "SOBI"
    n_sources: int = 5
    n_lags: int = 50

@dataclass
class FeatureFreqs:
    BPFO: Optional[float] = None
    BPFI: Optional[float] = None
    BSF: Optional[float] = None
    FTF: Optional[float] = None

@dataclass
class ExperimentConfig:
    name: str = "experiment"
    data_dir: str = "data"
    output_dir: str = "outputs"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tfa: TFAConfig = field(default_factory=TFAConfig)
    bss: BSSConfig = field(default_factory=BSSConfig)
    feature_freqs: FeatureFreqs = field(default_factory=FeatureFreqs)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从 YAML 文件加载配置"""
        ...
```

### 4.2 YAML 配置文件示例

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

---

## 5. 导入方式统一

### 5.1 当前问题

```python
# ❌ 每个脚本都需要这样
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import load_cwru
```

### 5.2 重构后

```python
# ✅ 直接导入包
from bss_test.io.cwru import load_cwru
from bss_test.preprocessing import preprocess_signals
from bss_test.tfa.factory import build_observation_matrix
from bss_test.bss.factory import bss_factory
from bss_test.evaluation import compute_independence_metric, plot_envelope_spectrum
from bss_test.utils.config import ExperimentConfig
from bss_test.utils.logger import get_logger
```

### 5.3 `__init__.py` 导出设计

```python
# src/bss_test/__init__.py
__version__ = "0.3.0"

from bss_test.io.cwru import load_cwru
from bss_test.preprocessing import preprocess_signals
from bss_test.tfa.factory import build_observation_matrix
from bss_test.bss.factory import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
)
from bss_test.utils.config import ExperimentConfig
```

---

## 6. 实验脚本重组

### 6.1 脚本映射关系

| 原文件 | 新位置 | 备注 |
|--------|--------|------|
| `main_cwru.py` | `experiments/single/cwru.py` | 重写使用新导入 |
| `main_phm_milling.py` | `experiments/single/phm_milling.py` | 重写使用新导入 |
| `main_nasa_milling.py` | `experiments/single/nasa_milling.py` | 重写使用新导入 |
| `run_bss_comparison.py` | `experiments/comparison/bss_methods.py` | 重写使用新导入 |
| `run_tfa_comparison.py` | `experiments/comparison/tfa_methods.py` | 重写使用新导入 |
| `run_full_comparison.py` + `_v2.py` | `experiments/comparison/full.py` | 合并，v2 改进作为可选参数 |
| `generate_comparison_report.py` | `experiments/reports/comparison_report.py` | 重写使用新导入 |
| `generate_summary.py` | `experiments/reports/summary.py` | 重写使用新导入 |

### 6.2 运行方式

```bash
# 方式 1：直接运行
python experiments/single/cwru.py

# 方式 2：模块方式运行
python -m experiments.single.cwru
```

### 6.3 `run_full_comparison.py` 合并策略

原 `run_full_comparison.py` 和 `_v2.py` 的区别：
- v1: 单次 train/test split
- v2: 重叠滑动窗口 + PCA + 交叉验证

合并方案：v2 的改进作为可选参数

```python
# experiments/comparison/full.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pca", action="store_true", help="使用 PCA 降维")
    parser.add_argument("--cv-folds", type=int, default=1, help="交叉验证折数 (1=单次分割)")
    parser.add_argument("--overlap", type=float, default=0.0, help="滑动窗口重叠率")
    args = parser.parse_args()
```

---

## 7. 不变的部分

- `preprocessing.py`、`feature_extractor.py`、`ml_classifier.py`、`evaluation.py` 保持单文件
- `tests/` 目录暂时保持现状
- `data/`、`outputs/` 目录结构不变

---

## 8. 实施顺序

1. 创建 `src/bss_test/` 包结构和 `__init__.py`
2. 迁移 `data_loader.py` → `io/` 子包
3. 迁移 `cwt_module.py` → `tfa/` 子包
4. 迁移 `bss_module.py` → `bss/` 子包
5. 迁移工具模块（config、logger、exceptions）→ `utils/`
6. 保留单文件模块（preprocessing、feature_extractor、ml_classifier、evaluation）
7. 创建 YAML 配置文件
8. 重组实验脚本到 `experiments/`
9. 更新 Makefile 和 pyproject.toml
10. 删除根目录旧脚本
