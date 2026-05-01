"""
深度学习分类器对比实验
用法: python -m experiments.comparison.dl_classifiers
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
from bss_test.feature_extractor import extract_time_domain_features, extract_freq_domain_features
from bss_test.evaluation import (
    plot_classifier_comparison,
    plot_confusion_matrix_grid,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

DL_METHODS = ["cnn", "lstm", "transformer"]


def load_cwru_dataset(config):
    """加载 CWRU 多故障类型数据，返回特征和标签"""
    from bss_test.io.cwru import load_cwru

    fault_types = ["normal", "inner_race_007", "ball_007", "outer_race_6_007"]
    all_features = []
    all_labels = []

    for fault_type in fault_types:
        try:
            signals, fs, rpm = load_cwru(
                data_dir=config.data_dir,
                fault_type=fault_type,
                load=0,
                channels=["DE"],
            )
            n_use = min(signals.shape[1], int(2.0 * fs))
            signals = signals[:, :n_use]

            preprocess_config = {
                "detrend": config.preprocess.detrend,
                "bandpass": config.preprocess.bandpass,
                "normalize": config.preprocess.normalize,
            }
            signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)

            seg_len = int(0.25 * fs_pre)
            step = seg_len // 2
            n_segments = (signals_pre.shape[1] - seg_len) // step + 1
            for seg_idx in range(n_segments):
                start = seg_idx * step
                seg = signals_pre[0, start:start + seg_len]
                feat_td = extract_time_domain_features(seg)
                feat_fd = extract_freq_domain_features(seg, fs_pre)
                features = np.concatenate([feat_td, feat_fd])
                all_features.append(features)
                all_labels.append(fault_type)

            logger.info(f"  {fault_type}: {n_segments} 段")
        except Exception as e:
            logger.warning(f"  {fault_type} 加载失败: {e}")

    return np.array(all_features), np.array(all_labels)


def load_phm_dataset(config):
    """加载 PHM 2010 数据，返回特征和标签"""
    from bss_test.io.phm import load_phm_cut

    all_features = []
    all_labels = []

    tool_labels = {1: "low_wear", 4: "medium_wear", 6: "high_wear"}

    for tool_id, label in tool_labels.items():
        try:
            signals, fs, cut_no = load_phm_cut(
                tool_id=tool_id,
                cut_no=150,
                data_dir=config.data_dir,
                sensor_types=["vib_x", "vib_y", "vib_z"],
            )
            n_use = min(signals.shape[1], int(1.0 * fs))
            signals = signals[:, :n_use]

            preprocess_config = {
                "detrend": config.preprocess.detrend,
                "bandpass": config.preprocess.bandpass,
                "normalize": config.preprocess.normalize,
            }
            signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)

            seg_len = int(0.2 * fs_pre)
            step = seg_len // 2
            n_segments = (signals_pre.shape[1] - seg_len) // step + 1
            for seg_idx in range(n_segments):
                start = seg_idx * step
                seg = signals_pre[0, start:start + seg_len]
                feat_td = extract_time_domain_features(seg)
                feat_fd = extract_freq_domain_features(seg, fs_pre)
                features = np.concatenate([feat_td, feat_fd])
                all_features.append(features)
                all_labels.append(label)

            logger.info(f"  c{tool_id} ({label}): {n_segments} 段")
        except Exception as e:
            logger.warning(f"  c{tool_id} 加载失败: {e}")

    return np.array(all_features), np.array(all_labels)


def run_dl_comparison(X, y, dataset_name, output_dir, n_epochs=50):
    """在给定数据集上运行深度学习分类器对比"""
    from sklearn.model_selection import train_test_split
    from bss_test.dl_classifier import train_dl_classifier, evaluate_dl_classifier

    logger.info(f"\n  数据形状: {X.shape}, 类别: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    logger.info(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    results_list = []

    for dl_method in DL_METHODS:
        logger.info(f"\n    --- {dl_method.upper()} ---")
        try:
            start_time = time.time()
            model_dict = train_dl_classifier(
                X_train, y_train,
                method=dl_method,
                n_epochs=n_epochs,
                batch_size=32,
                learning_rate=0.001,
            )
            elapsed = time.time() - start_time

            metrics = evaluate_dl_classifier(model_dict, X_test, y_test)

            results_list.append({
                "method": dl_method,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "confusion_matrix": metrics["confusion_matrix"],
                "label_names": metrics["label_names"],
                "time": elapsed,
            })

            logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"      F1-Macro: {metrics['f1_macro']:.4f}")
            logger.info(f"      时间: {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"      失败: {e}")

    # 生成柱状图
    if results_list:
        bar_data = [{"method": r["method"], "accuracy": r["accuracy"],
                     "f1_macro": r["f1_macro"]} for r in results_list]
        fig, _ = plot_classifier_comparison(bar_data, title_prefix=f"{dataset_name} DL ")
        fig.savefig(output_dir / f"{dataset_name}_dl_comparison.png", dpi=200)
        plt.close(fig)

        # 混淆矩阵
        fig, _ = plot_confusion_matrix_grid(results_list, title_prefix=f"{dataset_name} DL ")
        fig.savefig(output_dir / f"{dataset_name}_dl_confusion_matrix.png", dpi=200)
        plt.close(fig)

    return results_list


def main():
    """运行深度学习分类器对比实验"""
    setup_academic_style()

    output_dir = Path("outputs/dl_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("深度学习分类器对比实验")
    logger.info("=" * 60)
    logger.info(f"方法: {[m.upper() for m in DL_METHODS]}")

    all_report_rows = []

    # --- CWRU 数据集 ---
    logger.info("\n" + "=" * 60)
    logger.info("[1/2] CWRU 轴承故障分类")
    logger.info("=" * 60)
    cwru_config = ExperimentConfig.from_yaml("configs/cwru.yaml")
    X_cwru, y_cwru = load_cwru_dataset(cwru_config)

    if len(X_cwru) > 0:
        cwru_results = run_dl_comparison(X_cwru, y_cwru, "CWRU", output_dir)
        for r in cwru_results:
            all_report_rows.append({
                "dataset": "CWRU",
                "method": r["method"],
                "accuracy": f"{r['accuracy']:.4f}",
                "f1_macro": f"{r['f1_macro']:.4f}",
                "time": f"{r['time']:.2f}",
            })

    # --- PHM 2010 数据集 ---
    logger.info("\n" + "=" * 60)
    logger.info("[2/2] PHM 2010 磨损状态分类")
    logger.info("=" * 60)
    phm_config = ExperimentConfig.from_yaml("configs/phm2010.yaml")
    X_phm, y_phm = load_phm_dataset(phm_config)

    if len(X_phm) > 0:
        phm_results = run_dl_comparison(X_phm, y_phm, "PHM2010", output_dir)
        for r in phm_results:
            all_report_rows.append({
                "dataset": "PHM2010",
                "method": r["method"],
                "accuracy": f"{r['accuracy']:.4f}",
                "f1_macro": f"{r['f1_macro']:.4f}",
                "time": f"{r['time']:.2f}",
            })

    # 保存汇总 CSV
    csv_path = output_dir / "summary.csv"
    keys = ["dataset", "method", "accuracy", "f1_macro", "time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_report_rows)

    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
