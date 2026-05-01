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
    preprocess_signals,
    ExperimentConfig,
)
from bss_test.io.cwru import load_cwru
from bss_test.tfa import build_observation_matrix, cwt_transform
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_correlation_matrix,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """运行 CWRU 实验"""
    setup_academic_style()
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
    preprocess_config = {
        "detrend": config.preprocess.detrend,
        "bandpass": config.preprocess.bandpass,
        "normalize": config.preprocess.normalize,
    }
    signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)
    logger.info(f"  预处理后形状: {signals_pre.shape}, fs: {fs_pre} Hz")
    
    # 构建观测矩阵
    logger.info("\n[3/5] 构建观测矩阵...")
    cwt_config = {
        "mode": config.cwt.mode,
        "tfa_method": config.cwt.tfa_method,
        "n_bands": config.cwt.n_bands,
        "freq_range": config.cwt.freq_range,
        "wavelet": config.cwt.wavelet,
    }
    X, labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
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
    fig.savefig(output_dir / "envelope_spectrum.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_correlation_matrix(S_est, title="CWRU Source Correlation")
    fig.savefig(output_dir / "correlation_matrix.png", dpi=200)
    plt.close(fig)
    
    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
