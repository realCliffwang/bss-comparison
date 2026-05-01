"""
BSS 方法对比实验
用法: python -m experiments.comparison.bss_methods
"""

import os
from pathlib import Path
import numpy as np
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test import (
    preprocess_signals,
    ExperimentConfig,
)
from bss_test.io.cwru import load_cwru
from bss_test.tfa import build_observation_matrix
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_correlation_matrix,
    plot_bss_metrics_comparison,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# BSS 方法列表
BSS_METHODS = ["sobi", "fastica", "jade", "picard"]


def main():
    """运行 BSS 方法对比实验"""
    setup_academic_style()
    # 加载配置
    config = ExperimentConfig.from_yaml("configs/cwru.yaml")
    
    # 创建输出目录
    output_dir = Path(config.output_dir) / "bss_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("BSS 方法对比实验")
    logger.info("=" * 60)
    logger.info(f"方法: {[m.upper() for m in BSS_METHODS]}")
    
    # 加载数据
    logger.info("\n[1/4] 加载 CWRU 数据...")
    signals, fs, rpm = load_cwru(
        data_dir=config.data_dir,
        fault_type="inner_race_007",
        load=0,
        channels=["DE"],
    )
    logger.info(f"  加载 {signals.shape[0]} 通道, {signals.shape[1]} 样本 @ {fs} Hz")
    
    # 使用子集
    n_use = min(signals.shape[1], int(2.0 * fs))
    signals = signals[:, :n_use]
    logger.info(f"  使用前 {n_use} 样本 ({n_use/fs:.2f} 秒)")
    
    # 预处理
    logger.info("\n[2/4] 预处理...")
    preprocess_config = {
        "detrend": config.preprocess.detrend,
        "bandpass": config.preprocess.bandpass,
        "normalize": config.preprocess.normalize,
    }
    signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)
    logger.info(f"  预处理后形状: {signals_pre.shape}, fs: {fs_pre} Hz")
    
    # 构建观测矩阵
    logger.info("\n[3/4] 构建观测矩阵...")
    cwt_config = {
        "mode": config.cwt.mode,
        "tfa_method": config.cwt.tfa_method,
        "n_bands": config.cwt.n_bands,
        "freq_range": config.cwt.freq_range,
        "wavelet": config.cwt.wavelet,
    }
    X, labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
    logger.info(f"  观测矩阵: {X.shape[0]} 观测 × {X.shape[1]} 样本")
    
    # 运行 BSS 方法对比
    logger.info("\n[4/4] 运行 BSS 方法对比...")
    results = {}
    report_rows = []
    
    for method in BSS_METHODS:
        logger.info(f"\n  --- {method.upper()} ---")
        try:
            start_time = time.time()
            S_est, A_est, W = bss_factory(
                X, 
                method=method, 
                n_components=config.bss.n_sources
            )
            elapsed = time.time() - start_time
            
            # 计算指标
            indep = compute_independence_metric(S_est)
            ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)
            
            results[method] = {
                "S_est": S_est,
                "independence": indep,
                "ffds": ffds,
                "time": elapsed,
                "status": "OK",
            }
            
            logger.info(f"    独立性: {indep:.4f}")
            logger.info(f"    FFDS: {ffds:.2f}")
            logger.info(f"    时间: {elapsed:.2f}s")
            
            # 保存结果
            report_rows.append({
                "method": method,
                "independence": f"{indep:.4f}",
                "ffds": f"{ffds:.2f}",
                "time": f"{elapsed:.2f}",
                "status": "OK",
            })
            
            # 生成包络谱
            fig, _ = plot_envelope_spectrum(
                S_est, fs_pre,
                fault_freqs=config.feature_freqs,
                title_prefix=f"CWRU {method.upper()} — "
            )
            fig.savefig(output_dir / f"{method}_envelope_spectrum.png", dpi=200)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"    失败: {e}")
            results[method] = {
                "S_est": None,
                "independence": float("nan"),
                "ffds": float("nan"),
                "time": 0,
                "status": f"ERROR: {str(e)[:60]}",
            }
            report_rows.append({
                "method": method,
                "independence": "",
                "ffds": "",
                "time": "",
                "status": f"ERROR: {str(e)[:60]}",
            })
    
    # 生成对比图
    logger.info("\n  生成对比图...")
    fig, axes = plt.subplots(len(BSS_METHODS), 1, figsize=(14, 4 * len(BSS_METHODS)))
    
    for ax, method in zip(axes, BSS_METHODS):
        if results[method]["S_est"] is not None:
            S_est = results[method]["S_est"]
            sig = S_est[0]  # 使用第一个分离源
            
            # 计算包络谱
            from scipy.signal import hilbert
            analytic = hilbert(sig)
            envelope = np.abs(analytic)
            N = len(envelope)
            env_spec = np.abs(np.fft.rfft(envelope))
            freq = np.fft.rfftfreq(N, 1.0 / fs_pre)
            
            # 绘图
            ax.plot(freq, env_spec, linewidth=0.5, color="steelblue")
            ax.set_xlim(0, min(500, fs_pre / 2))
            ax.set_ylabel("Amplitude")
            
            # 标记故障频率
            colors = ["red", "green", "orange", "purple"]
            for i, (name, fval) in enumerate(config.feature_freqs.items()):
                if fval < fs_pre / 2:
                    ax.axvline(fval, color=colors[i % len(colors)], linestyle="--",
                               alpha=0.7, linewidth=1.5, label=f"{name}={fval:.1f}Hz")
            
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{method.upper()} — Independence={results[method]['independence']:.3f}, "
                        f"FFDS={results[method]['ffds']:.1f}")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{method.upper()}\n{results[method]['status']}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{method.upper()} — FAILED")
    
    axes[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    fig.savefig(output_dir / "bss_comparison_grid.png", dpi=200)
    plt.close(fig)

    # 生成指标柱状图
    logger.info("\n  生成指标柱状图...")
    metrics_for_bar = {}
    for method in BSS_METHODS:
        if results[method]["S_est"] is not None:
            metrics_for_bar[method] = {
                "independence": results[method]["independence"],
                "ffds": results[method]["ffds"],
            }
    if metrics_for_bar:
        fig, _ = plot_bss_metrics_comparison(metrics_for_bar, title_prefix="CWRU ")
        fig.savefig(output_dir / "bss_metrics_bar.png", dpi=200)
        plt.close(fig)

    # 保存 CSV 报告
    csv_path = output_dir / "summary.csv"
    keys = ["method", "independence", "ffds", "time", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report_rows)
    
    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
