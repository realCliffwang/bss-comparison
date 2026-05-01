"""
TFA 方法对比实验
用法: python -m experiments.comparison.tfa_methods
"""

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
    plot_tfa_metrics_comparison,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# TFA 方法列表
TFA_METHODS = ["cwt", "stft", "wpt"]


def main():
    """运行 TFA 方法对比实验"""
    setup_academic_style()
    config = ExperimentConfig.from_yaml("configs/cwru.yaml")

    output_dir = Path(config.output_dir) / "tfa_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TFA 方法对比实验")
    logger.info("=" * 60)
    logger.info(f"方法: {[m.upper() for m in TFA_METHODS]}")

    # 加载数据
    logger.info("\n[1/4] 加载 CWRU 数据...")
    signals, fs, rpm = load_cwru(
        data_dir=config.data_dir,
        fault_type="inner_race_007",
        load=0,
        channels=["DE"],
    )
    logger.info(f"  加载 {signals.shape[0]} 通道, {signals.shape[1]} 样本 @ {fs} Hz")

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

    # 运行 TFA 方法对比
    logger.info("\n[3/4] 运行 TFA 方法对比...")
    results = {}
    report_rows = []

    for tfa_method in TFA_METHODS:
        logger.info(f"\n  --- {tfa_method.upper()} ---")
        try:
            start_time = time.time()

            # 构建观测矩阵（WPT 使用不同的小波名）
            tfa_config = {
                "mode": config.cwt.mode,
                "tfa_method": tfa_method,
                "n_bands": config.cwt.n_bands,
                "freq_range": config.cwt.freq_range,
                "wavelet": "db4" if tfa_method == "wpt" else config.cwt.wavelet,
            }
            X, labels = build_observation_matrix(signals_pre, fs_pre, tfa_config)
            logger.info(f"    观测矩阵: {X.shape[0]} × {X.shape[1]}")

            # BSS 分离 (固定使用 SOBI)
            S_est, A_est, W = bss_factory(
                X,
                method="sobi",
                n_components=config.bss.n_sources,
            )

            elapsed = time.time() - start_time

            # 计算指标
            indep = compute_independence_metric(S_est)
            ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)

            results[tfa_method] = {
                "S_est": S_est,
                "independence": indep,
                "ffds": ffds,
                "time": elapsed,
                "n_obs": X.shape[0],
            }

            logger.info(f"    观测数: {X.shape[0]}")
            logger.info(f"    独立性: {indep:.4f}")
            logger.info(f"    FFDS: {ffds:.2f}")
            logger.info(f"    时间: {elapsed:.2f}s")

            report_rows.append({
                "method": tfa_method,
                "n_obs": X.shape[0],
                "independence": f"{indep:.4f}",
                "ffds": f"{ffds:.2f}",
                "time": f"{elapsed:.2f}",
                "status": "OK",
            })

            # 生成包络谱
            fig, _ = plot_envelope_spectrum(
                S_est, fs_pre,
                fault_freqs=config.feature_freqs,
                title_prefix=f"CWRU {tfa_method.upper()} + SOBI — ",
            )
            fig.savefig(output_dir / f"{tfa_method}_envelope_spectrum.png", dpi=200)
            plt.close(fig)

        except Exception as e:
            logger.error(f"    失败: {e}")
            results[tfa_method] = {
                "S_est": None,
                "independence": float("nan"),
                "ffds": float("nan"),
                "time": 0,
                "n_obs": 0,
            }
            report_rows.append({
                "method": tfa_method,
                "n_obs": "",
                "independence": "",
                "ffds": "",
                "time": "",
                "status": f"ERROR: {str(e)[:60]}",
            })

    # 生成柱状图
    logger.info("\n[4/4] 生成柱状图...")
    metrics_for_bar = {
        m: {"ffds": results[m]["ffds"]}
        for m in TFA_METHODS
        if results[m]["S_est"] is not None
    }
    if metrics_for_bar:
        fig, _ = plot_tfa_metrics_comparison(metrics_for_bar, title_prefix="CWRU ")
        fig.savefig(output_dir / "tfa_metrics_bar.png", dpi=200)
        plt.close(fig)

    # 保存 CSV
    csv_path = output_dir / "summary.csv"
    keys = ["method", "n_obs", "independence", "ffds", "time", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report_rows)

    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
