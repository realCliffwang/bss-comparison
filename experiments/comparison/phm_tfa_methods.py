"""
PHM 2010 TFA 方法对比实验
用法: python -m experiments.comparison.phm_tfa_methods
"""

from pathlib import Path
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test import (
    preprocess_signals,
    ExperimentConfig,
)
from bss_test.io.phm import load_phm_cut
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

TFA_METHODS = ["cwt", "stft", "wpt"]


def main():
    """运行 PHM 2010 TFA 方法对比实验"""
    setup_academic_style()
    config = ExperimentConfig.from_yaml("configs/phm2010.yaml")

    output_dir = Path(config.output_dir) / "tfa_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHM 2010 TFA 方法对比实验")
    logger.info("=" * 60)
    logger.info(f"方法: {[m.upper() for m in TFA_METHODS]}")

    all_report_rows = []
    tool_ids = [1, 2, 3, 4, 5, 6]

    for tool_id in tool_ids:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"刀具 c{tool_id}")
        logger.info("=" * 60)

        # 加载数据
        try:
            signals, fs, cut_no = load_phm_cut(
                tool_id=tool_id,
                cut_no=150,
                data_dir=config.data_dir,
                sensor_types=["vib_x", "vib_y", "vib_z"],
            )
        except Exception as e:
            logger.warning(f"c{tool_id} 加载失败: {e}")
            continue

        n_use = min(signals.shape[1], int(1.0 * fs))
        signals = signals[:, :n_use]

        # 预处理
        preprocess_config = {
            "detrend": config.preprocess.detrend,
            "bandpass": config.preprocess.bandpass,
            "normalize": config.preprocess.normalize,
        }
        signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)

        # 运行 TFA 方法对比
        results = {}
        report_rows = []

        for tfa_method in TFA_METHODS:
            logger.info(f"\n  --- {tfa_method.upper()} ---")
            try:
                start_time = time.time()

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

                indep = compute_independence_metric(S_est)
                ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)

                results[tfa_method] = {
                    "S_est": S_est,
                    "independence": indep,
                    "ffds": ffds,
                    "time": elapsed,
                    "n_obs": X.shape[0],
                }

                logger.info(f"    独立性: {indep:.4f}")
                logger.info(f"    FFDS: {ffds:.2f}")
                logger.info(f"    时间: {elapsed:.2f}s")

                report_rows.append({
                    "tool": f"c{tool_id}",
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
                    title_prefix=f"PHM c{tool_id} {tfa_method.upper()} + SOBI — ",
                )
                fig.savefig(output_dir / f"c{tool_id}_{tfa_method}_envelope_spectrum.png", dpi=200)
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
                    "tool": f"c{tool_id}",
                    "method": tfa_method,
                    "n_obs": "",
                    "independence": "",
                    "ffds": "",
                    "time": "",
                    "status": f"ERROR: {str(e)[:60]}",
                })

        # 生成柱状图
        metrics_for_bar = {
            m: {"ffds": results[m]["ffds"]}
            for m in TFA_METHODS
            if results[m]["S_est"] is not None
        }
        if metrics_for_bar:
            fig, _ = plot_tfa_metrics_comparison(metrics_for_bar, title_prefix=f"PHM c{tool_id} ")
            fig.savefig(output_dir / f"c{tool_id}_tfa_metrics_bar.png", dpi=200)
            plt.close(fig)

        all_report_rows.extend(report_rows)

    # 保存 CSV
    csv_path = output_dir / "summary.csv"
    keys = ["tool", "method", "n_obs", "independence", "ffds", "time", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_report_rows)

    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
