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
- **支持数据集**: CWRU, PHM 2010, NASA Milling

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行实验
```bash
# CWRU轴承故障检测
python main_cwru.py

# PHM 2010铣削刀具磨损
python main_phm_milling.py

# NASA铣削数据集
python main_nasa_milling.py

# 生成对比报告
python generate_comparison_report.py
```

### 查看结果
实验结果保存在 `outputs/` 目录，包含可视化图表和CSV摘要。

## 项目结构

```
BSS-test/
├── main_cwru.py              # CWRU轴承实验
├── main_phm_milling.py       # PHM铣削实验
├── main_nasa_milling.py      # NASA铣削实验
├── generate_comparison_report.py  # 对比报告生成
├── src/
│   ├── data_loader.py        # 数据加载
│   ├── preprocessing.py      # 信号预处理
│   ├── cwt_module.py         # 时频分析
│   ├── bss_module.py         # 盲源分离
│   ├── feature_extractor.py  # 特征提取
│   ├── ml_classifier.py      # 机器学习分类器
│   ├── evaluation.py         # 评估指标
│   └── utils.py              # 工具函数
├── data/                     # 数据集目录
├── outputs/                  # 实验结果
└── tests/                    # 测试文件
```

## 评估指标

### BSS质量指标
- **源独立性**: 平均绝对非对角相关系数 (越低越好)
- **FFDS**: 故障频率检测分数 (越高越好)
- **SIR**: 信号干扰比

### 分类指标
- **Accuracy**: 分类准确率
- **F1-Macro**: 宏平均F1分数

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
