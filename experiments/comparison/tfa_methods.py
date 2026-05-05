"""
TFA 方法对比实验
用法: python -m experiments.comparison.tfa_methods
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test.tfa import build_observation_matrix
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_tfa_metrics_comparison,
)
from bss_test.utils.logger import setup_logging

from experiments._common import load_cwru_signals, write_csv_summary, init_experiment

TFA_METHODS = ["cwt", "stft", "wpt"]


def main():
    config, output_dir, logger = init_experiment(
        "TFA 方法对比实验", "tfa_comparison", "configs/cwru.yaml",
    )
    logger.info(f"方法: {[m.upper() for m in TFA_METHODS]}")

    logger.info("\n[1/3] 加载 CWRU 数据...")
    signals_pre, fs_pre = load_cwru_signals(config, "inner_race_007", max_duration=2.0)
    logger.info(f"  预处理后: {signals_pre.shape}, fs: {fs_pre} Hz")

    logger.info("\n[2/3] 运行 TFA 方法对比...")
    results = {}
    report_rows = []

    for tfa_method in TFA_METHODS:
        logger.info(f"\n  --- {tfa_method.upper()} ---")
        try:
            start_time = time.time()

            tfa_config = {
                "mode": config.tfa.mode,
                "tfa_method": tfa_method,
                "n_bands": config.tfa.n_bands,
                "freq_range": config.tfa.freq_range,
                "wavelet": "db4" if tfa_method == "wpt" else config.tfa.wavelet,
            }
            X, labels = build_observation_matrix(signals_pre, fs_pre, tfa_config)

            S_est, A_est, W = bss_factory(X, method="sobi", n_components=config.bss.n_sources)
            elapsed = time.time() - start_time

            indep = compute_independence_metric(S_est)
            ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)

            results[tfa_method] = {"S_est": S_est, "ffds": ffds}

            logger.info(f"    观测数: {X.shape[0]}")
            logger.info(f"    独立性: {indep:.4f}")
            logger.info(f"    FFDS: {ffds:.2f}")
            logger.info(f"    时间: {elapsed:.2f}s")

            report_rows.append({
                "method": tfa_method, "n_obs": X.shape[0],
                "independence": f"{indep:.4f}", "ffds": f"{ffds:.2f}",
                "time": f"{elapsed:.2f}", "status": "OK",
            })

            fig, _ = plot_envelope_spectrum(
                S_est, fs_pre, fault_freqs=config.feature_freqs,
                title_prefix=f"CWRU {tfa_method.upper()} + SOBI — ",
            )
            fig.savefig(output_dir / f"{tfa_method}_envelope_spectrum.png", dpi=200)
            plt.close(fig)

        except Exception as e:
            logger.error(f"    Failed: {e}")
            report_rows.append({
                "method": tfa_method, "n_obs": "", "independence": "",
                "ffds": "", "time": "", "status": f"ERROR: {e}",
            })

    logger.info("\n[3/3] 生成柱状图...")
    metrics_for_bar = {m: {"ffds": v["ffds"]} for m, v in results.items()}
    if metrics_for_bar:
        fig, _ = plot_tfa_metrics_comparison(metrics_for_bar, title_prefix="CWRU ")
        fig.savefig(output_dir / "tfa_metrics_bar.png", dpi=200)
        plt.close(fig)

    write_csv_summary(output_dir, report_rows)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
