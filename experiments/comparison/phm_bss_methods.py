"""
PHM 2010 BSS 方法对比实验
用法: python -m experiments.comparison.phm_bss_methods
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
    plot_bss_metrics_comparison,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

BSS_METHODS = ["sobi", "fastica", "jade", "picard"]


def main():
    """运行 PHM 2010 BSS 方法对比实验"""
    setup_academic_style()
    config = ExperimentConfig.from_yaml("configs/phm2010.yaml")

    output_dir = Path(config.output_dir) / "bss_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHM 2010 BSS 方法对比实验")
    logger.info("=" * 60)
    logger.info(f"方法: {[m.upper() for m in BSS_METHODS]}")

    all_results = {}
    all_report_rows = []

    # 遍历全部 6 个刀具
    tool_ids = [1, 2, 3, 4, 5, 6]

    for tool_id in tool_ids:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"刀具 c{tool_id}")
        logger.info("=" * 60)

        # 加载数据（取中间 cut，避免首尾异常）
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
        logger.info(f"  加载 {signals.shape[0]} 通道, {signals.shape[1]} 样本 @ {fs} Hz")

        # 预处理
        preprocess_config = {
            "detrend": config.preprocess.detrend,
            "bandpass": config.preprocess.bandpass,
            "normalize": config.preprocess.normalize,
        }
        signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)

        # 构建观测矩阵
        cwt_config = {
            "mode": config.cwt.mode,
            "tfa_method": config.cwt.tfa_method,
            "n_bands": config.cwt.n_bands,
            "freq_range": config.cwt.freq_range,
            "wavelet": config.cwt.wavelet,
        }
        X, labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
        logger.info(f"  观测矩阵: {X.shape[0]} 观测 × {X.shape[1]} 样本")

        # 运行 BSS 方法
        results = {}
        report_rows = []

        for method in BSS_METHODS:
            logger.info(f"\n  --- {method.upper()} ---")
            try:
                start_time = time.time()
                S_est, A_est, W = bss_factory(
                    X,
                    method=method,
                    n_components=config.bss.n_sources,
                )
                elapsed = time.time() - start_time

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

                report_rows.append({
                    "tool": f"c{tool_id}",
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
                    title_prefix=f"PHM c{tool_id} {method.upper()} — ",
                )
                fig.savefig(output_dir / f"c{tool_id}_{method}_envelope_spectrum.png", dpi=200)
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
                    "tool": f"c{tool_id}",
                    "method": method,
                    "independence": "",
                    "ffds": "",
                    "time": "",
                    "status": f"ERROR: {str(e)[:60]}",
                })

        # 生成指标柱状图
        metrics_for_bar = {}
        for method in BSS_METHODS:
            if results[method]["S_est"] is not None:
                metrics_for_bar[method] = {
                    "independence": results[method]["independence"],
                    "ffds": results[method]["ffds"],
                }
        if metrics_for_bar:
            fig, _ = plot_bss_metrics_comparison(metrics_for_bar, title_prefix=f"PHM c{tool_id} ")
            fig.savefig(output_dir / f"c{tool_id}_bss_metrics_bar.png", dpi=200)
            plt.close(fig)

        all_results[tool_id] = results
        all_report_rows.extend(report_rows)

    # 保存 CSV 报告
    csv_path = output_dir / "summary.csv"
    keys = ["tool", "method", "independence", "ffds", "time", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_report_rows)

    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
