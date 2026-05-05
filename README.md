# BSS-Test: 轴承故障诊断盲源分离框架

基于盲源分离 (BSS) 和时频分析 (TFA) 的轴承故障诊断综合框架。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目概述

本项目实现了完整的振动信号故障诊断流程：

```
原始信号 → 预处理 → 时频分析 → 盲源分离 → 特征提取 → 分类诊断
```

### 核心功能
- **7种时频分析方法**: CWT, STFT, WPT, VMD, EMD, EEMD, CEEMDAN
- **6种BSS算法**: SOBI, FastICA, JADE, PICARD, NMF, PCA
- **5种ML分类器**: SVM, Random Forest, XGBoost, KNN, LDA
- **3种DL分类器**: 1D-CNN, LSTM, Transformer（可选 PyTorch）
- **WDCNN 端到端分类器**: 直接处理原始振动信号，无需特征提取（可选 PyTorch）
- **支持数据集**: CWRU, PHM 2010, NASA Milling
- **报告生成**: 自包含 HTML 和 Markdown 实验报告

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行所有测试（192 个）
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=src --cov-report=html
```

### 运行实验

所有实验脚本使用模块方式运行：

```bash
# 设置 PYTHONPATH（如未安装为包）
$env:PYTHONPATH = "src"   # PowerShell
export PYTHONPATH=src      # Linux/Mac

# 单数据集实验
python -m experiments.single.cwru           # CWRU 轴承故障检测
python -m experiments.single.phm_milling    # PHM 2010 铣削刀具磨损
python -m experiments.single.nasa_milling   # NASA 铣削数据集

# 对比实验
python -m experiments.comparison.bss_methods    # BSS 方法对比（SOBI/FastICA/JADE/PICARD）
python -m experiments.comparison.tfa_methods    # TFA 方法对比（CWT/STFT/WPT）
python -m experiments.comparison.ml_classifiers # ML 分类器对比（SVM/RF/KNN/LDA）
python -m experiments.comparison.phm_bss_methods   # PHM BSS 方法对比
python -m experiments.comparison.phm_tfa_methods   # PHM TFA 方法对比
python -m experiments.comparison.phm_classifiers   # PHM 分类器对比
python -m experiments.comparison.dl_classifiers    # DL 分类器对比（需 PyTorch）
python -m experiments.comparison.wdcnn_vs_traditional  # WDCNN vs 传统方法（需 PyTorch）
python -m experiments.comparison.wdcnn_vs_bss      # WDCNN vs BSS 对比（需 PyTorch）
```

### 查看结果
实验结果保存在 `outputs/` 目录，包含可视化图表（PNG, DPI=200）和 CSV 摘要。

### 实验结果摘要

详见 [`outputs/experiment_summary.md`](outputs/experiment_summary.md)。

| 场景 | 推荐方法 | 指标 |
|------|----------|------|
| 有标签 + 小样本 (<100) | RF / LDA | 100% accuracy, 0.01s |
| 有标签 + 大样本 | WDCNN | 96~99% accuracy |
| 无标签 + 故障检测 | JADE + STFT | FFDS 最高 (10.94 / 14.46) |
| 速度优先 | LDA | 0s 训练 |

### 生成报告

```python
from bss_test.report import ExperimentReport

report = ExperimentReport("实验报告", output_dir="outputs/my_experiment")
report.add_text("实验概述", "使用 CWRU 轴承数据...")
report.add_figure(fig, "包络谱对比")
report.add_metrics_table(results, "分类器性能")
report.to_html()      # → outputs/my_experiment/report.html
report.to_markdown()  # → outputs/my_experiment/report.md
```

## 项目结构

```
BSS-test/
├── configs/                        # YAML 配置文件
│   ├── default.yaml
│   ├── cwru.yaml
│   ├── phm2010.yaml
│   └── nasa.yaml
│
├── src/
│   └── bss_test/                   # 主包
│       ├── __init__.py             # 包入口，re-export 常用函数
│       ├── types.py                # 类型定义
│       ├── preprocessing.py        # 信号预处理
│       ├── feature_extractor.py    # 特征提取
│       ├── ml_classifier.py        # ML 分类器
│       ├── dl_classifier.py        # DL 分类器（CNN/LSTM/Transformer）
│       ├── wdcnn.py                # WDCNN 端到端原始信号分类器
│       ├── _torch_training.py      # 共享 PyTorch 训练循环
│       ├── metrics.py              # 评估指标（独立性、FFDS、SIR、evaluate_bss）
│       ├── visualization.py        # 绘图函数（学术论文风格）
│       ├── evaluation.py           # 向后兼容 re-export（metrics + visualization）
│       ├── report.py               # 报告生成（HTML/Markdown）
│       ├── io/                     # 数据 I/O（cwru/phm/nasa）
│       ├── tfa/                    # 时频分析（cwt/stft/wpt/emd）
│       ├── bss/                    # 盲源分离（sobi/ica/jade）
│       └── utils/                  # 工具（config_types/config_io/logger/exceptions/synthetic）
│
├── experiments/
│   ├── _common.py                  # 共享工具函数
│   ├── single/                     # 单数据集实验
│   │   ├── cwru.py
│   │   ├── phm_milling.py
│   │   └── nasa_milling.py
│   ├── comparison/                 # 对比实验
│   │   ├── bss_methods.py
│   │   ├── tfa_methods.py
│   │   ├── ml_classifiers.py
│   │   ├── phm_bss_methods.py
│   │   ├── phm_tfa_methods.py
│   │   ├── phm_classifiers.py
│   │   ├── dl_classifiers.py
│   │   ├── wdcnn_vs_traditional.py
│   │   └── wdcnn_vs_bss.py
│   └── reports/                    # 报告生成
│
├── data/                           # 数据集目录（gitignore）
├── outputs/                        # 实验结果（gitignore）
└── tests/                          # 测试文件（192 个）
```

## 评估指标

### BSS质量指标
- **源独立性**: 平均绝对非对角相关系数 (越低越好)
- **FFDS**: 故障频率检测分数 (越高越好)
- **SIR**: 信号干扰比

### 分类指标
- **Accuracy**: 分类准确率
- **F1-Macro**: 宏平均F1分数

## 配置管理

实验配置通过 YAML 文件管理（`configs/` 目录），支持以下参数：
- `preprocess`: 预处理参数（去趋势、带通滤波、归一化）
- `tfa`: 时频分析参数（方法、小波、频带数、频率范围）
- `bss`: BSS 参数（方法、源数量、时滞数）
- `feature_freqs`: 故障特征频率（BPFO/BPFI/BSF）

配置加载：`ExperimentConfig.from_yaml("configs/cwru.yaml")`，TFA 配置字段为 `config.tfa`。

配置模块已拆分为 `config_types.py`（dataclass 定义）和 `config_io.py`（加载/保存/CLI），`config.py` 为向后兼容 re-export 层。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
