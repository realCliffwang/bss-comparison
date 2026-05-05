"""
PHM 2010 BSS 方法对比实验
用法: python -m experiments.comparison.phm_bss_methods
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
    plot_bss_metrics_comparison,
)
from bss_test.utils.logger import setup_logging

from experiments._common import load_phm_signals, write_csv_summary, init_experiment

BSS_METHODS = ["sobi", "fastica", "jade", "picard"]


def main():
    config, output_dir, logger = init_experiment(
        "PHM 2010 BSS 方法对比实验", "bss_comparison", "configs/phm2010.yaml",
    )
    logger.info(f"方法: {[m.upper() for m in BSS_METHODS]}")

    all_report_rows = []

    for tool_id in [1, 2, 3, 4, 5, 6]:
        logger.info(f"\n{'='*60}")
        logger.info(f"刀具 c{tool_id}")
        logger.info("=" * 60)

        try:
            signals_pre, fs_pre = load_phm_signals(config, tool_id, cut_no=150, max_duration=1.0)
        except Exception as e:
            logger.warning(f"c{tool_id} failed: {e}")
            continue

        cwt_config = {
            "mode": config.tfa.mode,
            "tfa_method": config.tfa.tfa_method,
            "n_bands": config.tfa.n_bands,
            "freq_range": config.tfa.freq_range,
            "wavelet": config.tfa.wavelet,
        }
        X, labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
        logger.info(f"  观测矩阵: {X.shape[0]} x {X.shape[1]}")

        results = {}
        for method in BSS_METHODS:
            logger.info(f"\n  --- {method.upper()} ---")
            try:
                start_time = time.time()
                S_est, A_est, W = bss_factory(X, method=method, n_components=config.bss.n_sources)
                elapsed = time.time() - start_time

                indep = compute_independence_metric(S_est)
                ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)
                results[method] = {"independence": indep, "ffds": ffds}

                logger.info(f"    独立性: {indep:.4f}")
                logger.info(f"    FFDS: {ffds:.2f}")
                logger.info(f"    时间: {elapsed:.2f}s")

                all_report_rows.append({
                    "tool": f"c{tool_id}", "method": method,
                    "independence": f"{indep:.4f}", "ffds": f"{ffds:.2f}",
                    "time": f"{elapsed:.2f}", "status": "OK",
                })

                fig, _ = plot_envelope_spectrum(
                    S_est, fs_pre, fault_freqs=config.feature_freqs,
                    title_prefix=f"PHM c{tool_id} {method.upper()} — ",
                )
                fig.savefig(output_dir / f"c{tool_id}_{method}_envelope.png", dpi=200)
                plt.close(fig)

            except Exception as e:
                logger.error(f"    Failed: {e}")
                all_report_rows.append({
                    "tool": f"c{tool_id}", "method": method,
                    "independence": "", "ffds": "", "time": "", "status": f"ERROR: {e}",
                })

        if results:
            fig, _ = plot_bss_metrics_comparison(results, title_prefix=f"PHM c{tool_id} ")
            fig.savefig(output_dir / f"c{tool_id}_bss_metrics_bar.png", dpi=200)
            plt.close(fig)

    write_csv_summary(output_dir, all_report_rows)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
