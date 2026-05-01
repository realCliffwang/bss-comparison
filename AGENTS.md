# AGENTS.md — BSS-Test AI 协作指南

## 项目概述

轴承故障诊断框架，基于盲源分离 (BSS) + 时频分析 (TFA) + 机器学习/深度学习分类。

## 运行方式

```bash
# PYTHONPATH 方式运行（未安装为包时）
$env:PYTHONPATH = "src"
python -m experiments.single.cwru
```

未通过 pip 安装，使用 PYTHONPATH 指向 `src/`。

## 代码结构

- `src/bss_test/` — 主包，子包：`io/`、`tfa/`、`bss/`、`utils/`
- `src/bss_test/dl_classifier.py` — 深度学习分类器（CNN/LSTM/Transformer），可选 PyTorch
- `experiments/` — 实验脚本，`single/` 单数据集，`comparison/` 对比实验
- `configs/` — YAML 配置文件
- `outputs/` — 实验输出（gitignore）

## 对比实验脚本

| 脚本 | 数据集 | 对比内容 |
|------|--------|----------|
| `comparison/bss_methods.py` | CWRU | SOBI/FastICA/JADE/PICARD |
| `comparison/tfa_methods.py` | CWRU | CWT/STFT/WPT |
| `comparison/ml_classifiers.py` | CWRU+PHM | SVM/RF/KNN/LDA |
| `comparison/phm_bss_methods.py` | PHM 2010 | SOBI/FastICA/JADE/PICARD |
| `comparison/phm_tfa_methods.py` | PHM 2010 | CWT/STFT/WPT |
| `comparison/phm_classifiers.py` | PHM 2010 | SVM/RF/KNN/LDA |
| `comparison/dl_classifiers.py` | CWRU+PHM | CNN/LSTM/Transformer |

## 关键约定

1. **绘图风格**：`evaluation.py` 中的 `setup_academic_style()` 提供学术论文风格（白底、清晰网格、serif 字体）。所有绘图函数内部自动调用。
2. **DPI**：所有图表保存使用 `dpi=200`。
3. **配置系统**：`ExperimentConfig.from_yaml("configs/xxx.yaml")` 加载配置，dataclass 结构在 `utils/config.py`。
4. **包络谱标注**：故障频率使用图例标注（不直接写在图上），y 轴 98th percentile 自适应。
5. **WPT 小波**：WPT 使用 `db4`（不是 CWT 的 `cmor1.5-1.0`）。
6. **XGBoost**：可选依赖，`ml_classifiers.py` 会自动检测是否安装。
7. **PyTorch**：可选依赖，`dl_classifier.py` 会自动检测是否安装。DL 对比实验需设置 `KMP_DUPLICATE_LIB_OK=TRUE`（Anaconda 环境）。
8. **ML/DL 统一接口**：`train_classifier()` 和 `train_dl_classifier()` 返回格式一致（`{accuracy, f1_macro, confusion_matrix, predictions, true_labels, label_names}`）。

## 环境

- Python 3.8+
- 依赖：numpy, scipy, matplotlib, pywt, scikit-learn, pyyaml
- 可选：xgboost, torch>=2.0.0
