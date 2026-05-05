"""
BSS 方法对比实验
用法: python -m experiments.comparison.bss_methods
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from bss_test import ExperimentConfig
from bss_test.tfa import build_observation_matrix
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_bss_metrics_comparison,
)
from bss_test.utils.logger import setup_logging

from experiments._common import load_cwru_signals, write_csv_summary, init_experiment

BSS_METHODS = ["sobi", "fastica", "jade", "picard"]


def main():
    config, output_dir, logger = init_experiment(
        "BSS 方法对比实验", "bss_comparison", "configs/cwru.yaml",
    )
    logger.info(f"方法: {[m.upper() for m in BSS_METHODS]}")

    # 加载数据
    logger.info("\n[1/3] 加载 CWRU 数据...")
    signals_pre, fs_pre = load_cwru_signals(config, "inner_race_007", max_duration=2.0)
    logger.info(f"  预处理后: {signals_pre.shape}, fs: {fs_pre} Hz")

    # 构建观测矩阵
    logger.info("\n[2/3] 构建观测矩阵...")
    cwt_config = {
        "mode": config.tfa.mode,
        "tfa_method": config.tfa.tfa_method,
        "n_bands": config.tfa.n_bands,
        "freq_range": config.tfa.freq_range,
        "wavelet": config.tfa.wavelet,
    }
    X, labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
    logger.info(f"  观测矩阵: {X.shape[0]} x {X.shape[1]}")

    # BSS 方法对比
    logger.info("\n[3/3] 运行 BSS 方法对比...")
    results = {}
    report_rows = []

    for method in BSS_METHODS:
        logger.info(f"\n  --- {method.upper()} ---")
        try:
            start_time = time.time()
            S_est, A_est, W = bss_factory(X, method=method, n_components=config.bss.n_sources)
            elapsed = time.time() - start_time

            indep = compute_independence_metric(S_est)
            ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)

            results[method] = {"S_est": S_est, "independence": indep, "ffds": ffds}

            logger.info(f"    独立性: {indep:.4f}")
            logger.info(f"    FFDS: {ffds:.2f}")
            logger.info(f"    时间: {elapsed:.2f}s")

            report_rows.append({
                "method": method,
                "independence": f"{indep:.4f}",
                "ffds": f"{ffds:.2f}",
                "time": f"{elapsed:.2f}",
                "status": "OK",
            })

            fig, _ = plot_envelope_spectrum(
                S_est, fs_pre, fault_freqs=config.feature_freqs,
                title_prefix=f"CWRU {method.upper()} — ",
            )
            fig.savefig(output_dir / f"{method}_envelope_spectrum.png", dpi=200)
            plt.close(fig)

        except Exception as e:
            logger.error(f"    Failed: {e}")
            report_rows.append({
                "method": method, "independence": "", "ffds": "",
                "time": "", "status": f"ERROR: {e}",
            })

    # 指标柱状图
    metrics_for_bar = {m: {"independence": v["independence"], "ffds": v["ffds"]}
                       for m, v in results.items()}
    if metrics_for_bar:
        fig, _ = plot_bss_metrics_comparison(metrics_for_bar, title_prefix="CWRU ")
        fig.savefig(output_dir / "bss_metrics_bar.png", dpi=200)
        plt.close(fig)

    write_csv_summary(output_dir, report_rows)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
