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
- **支持数据集**: CWRU, PHM 2010, NASA Milling

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
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
```

### 查看结果
实验结果保存在 `outputs/` 目录，包含可视化图表（PNG, DPI=200）和 CSV 摘要。

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
│       ├── __init__.py
│       ├── types.py                # 类型定义
│       ├── preprocessing.py        # 信号预处理
│       ├── feature_extractor.py    # 特征提取
│       ├── ml_classifier.py        # ML 分类器
│       ├── dl_classifier.py        # DL 分类器（CNN/LSTM/Transformer）
│       ├── evaluation.py           # 评估指标 + 可视化
│       ├── io/                     # 数据 I/O（cwru/phm/nasa）
│       ├── tfa/                    # 时频分析（cwt/stft/wpt/emd）
│       ├── bss/                    # 盲源分离（sobi/ica/jade）
│       └── utils/                  # 工具（config/logger/exceptions）
│
├── experiments/
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
│   │   └── dl_classifiers.py
│   └── reports/                    # 报告生成
│
├── data/                           # 数据集目录（gitignore）
├── outputs/                        # 实验结果（gitignore）
└── tests/                          # 测试文件
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

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
