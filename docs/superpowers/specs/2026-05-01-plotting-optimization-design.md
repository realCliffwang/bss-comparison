# BSS-Test 绘图优化设计文档

> **日期:** 2026-05-01
> **版本:** 1.0
> **状态:** 已批准

---

## 1. 问题描述

当前 `evaluation.py` 中的绘图函数存在以下问题：

1. **视觉质量差** — 使用 matplotlib 默认样式，字体小、无统一配色
2. **信息呈现不清** — 故障频率标注线（红色虚线）和包络谱数据线重叠，文字标注直接写在图上导致看不清
3. **图表生成不正确** — 部分图片的内容和指示条混在一起

## 2. 设计目标

- 学术论文风格：白底、清晰网格、现代感
- 故障频率标注不与数据线重叠
- 字体 12pt+ 清晰可读
- x/y 轴范围自适应
- 输出 DPI 提升到 200

## 3. 方案选择

采用 **集中式样式系统** 方案：在 `evaluation.py` 顶部添加 `setup_academic_style()` 全局样式函数，所有绘图函数统一调用。

## 4. 改动范围

### 4.1 核心改动文件

| 文件 | 改动内容 |
|------|----------|
| `src/bss_test/evaluation.py` | 添加全局样式函数，修复所有绘图函数 |
| `experiments/single/cwru.py` | DPI 200，调用 setup_academic_style() |
| `experiments/single/phm_milling.py` | DPI 200，调用 setup_academic_style() |
| `experiments/single/nasa_milling.py` | DPI 200，调用 setup_academic_style() |
| `experiments/comparison/bss_methods.py` | DPI 200，调用 setup_academic_style() |

### 4.2 全局样式配置

```python
ACADEMIC_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
}
```

### 4.3 包络谱修复 (`plot_envelope_spectrum`)

- 标注文字从图上移到图例（`ax.legend()`），避免遮挡数据
- y 轴使用 98th percentile 自适应，留 20% 空间给标注
- x 轴根据故障频率最大值的 2.5 倍自适应

### 4.4 相关矩阵修复 (`plot_correlation_matrix`)

- 单源时显示 1×1 热力图（值为 1.0），而非纯文字

### 4.5 波形对比修复 (`plot_waveform_comparison`)

- offset 改用 `1.5 * max(abs(signal))`，确保信号不重叠

### 4.6 BSS 对比修复 (`plot_bss_comparison`)

- 同一行共享 y 轴范围，便于横向比较不同 BSS 方法

### 4.7 质量报告修复 (`plot_separation_quality_report`)

- 改用 `GridSpec` 布局，替代手动 `fig.add_axes()`

## 5. 输出清理与重跑

1. 删除 `outputs/` 下所有内容
2. 按顺序重跑所有实验脚本
3. 验证标准：标注不重叠、字体清晰、配色统一、轴范围自适应

## 6. 验证清单

- [ ] `plot_envelope_spectrum` — 故障频率标注在图例中，不遮挡数据
- [ ] `plot_correlation_matrix` — 单源时显示热力图
- [ ] `plot_waveform_comparison` — 信号不重叠
- [ ] `plot_bss_comparison` — 同行共享 y 轴
- [ ] `plot_separation_quality_report` — GridSpec 布局正常
- [ ] 所有图表 DPI 200，字体 12pt+
- [ ] 所有实验脚本可正常运行并生成输出
