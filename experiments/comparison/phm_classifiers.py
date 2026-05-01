"""
PHM 2010 分类器对比实验
用法: python -m experiments.comparison.phm_classifiers
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
from bss_test.io.phm import load_phm_cut
from bss_test.feature_extractor import extract_time_domain_features, extract_freq_domain_features
from bss_test.ml_classifier import train_classifier, evaluate_classifier
from bss_test.evaluation import (
    plot_classifier_comparison,
    plot_confusion_matrix_grid,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

CLASSIFIERS = ["svm", "rf", "knn", "lda"]

try:
    import xgboost
    CLASSIFIERS.insert(2, "xgb")
except ImportError:
    pass


def load_phm_dataset(config):
    """加载 PHM 2010 全部 6 刀具数据，返回特征和标签"""
    all_features = []
    all_labels = []

    # c1-c3 低磨损, c4 中磨损, c5-c6 高磨损
    tool_labels = {
        1: "low_wear",
        2: "low_wear",
        3: "low_wear",
        4: "medium_wear",
        5: "high_wear",
        6: "high_wear",
    }

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

            # 分段提取特征（每段 0.2 秒，重叠 50%）
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


def main():
    """运行 PHM 2010 分类器对比实验"""
    setup_academic_style()

    output_dir = Path("outputs/phm2010/classifier_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHM 2010 分类器对比实验")
    logger.info("=" * 60)
    logger.info(f"分类器: {[c.upper() for c in CLASSIFIERS]}")

    from sklearn.model_selection import train_test_split

    # 加载数据
    config = ExperimentConfig.from_yaml("configs/phm2010.yaml")
    logger.info("\n加载 PHM 2010 数据...")
    X, y = load_phm_dataset(config)

    if len(X) == 0:
        logger.error("无数据，退出")
        return

    logger.info(f"\n数据形状: {X.shape}, 类别: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    logger.info(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 运行分类器对比
    results_list = []
    report_rows = []

    for clf_method in CLASSIFIERS:
        logger.info(f"\n  --- {clf_method.upper()} ---")
        try:
            start_time = time.time()
            model = train_classifier(X_train, y_train, method=clf_method)
            elapsed = time.time() - start_time

            metrics = evaluate_classifier(model, X_test, y_test)

            results_list.append({
                "method": clf_method,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "confusion_matrix": metrics["confusion_matrix"],
                "label_names": metrics["label_names"],
                "time": elapsed,
            })

            report_rows.append({
                "method": clf_method,
                "accuracy": f"{metrics['accuracy']:.4f}",
                "f1_macro": f"{metrics['f1_macro']:.4f}",
                "time": f"{elapsed:.2f}",
                "status": "OK",
            })

            logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"      F1-Macro: {metrics['f1_macro']:.4f}")
            logger.info(f"      时间: {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"      失败: {e}")
            report_rows.append({
                "method": clf_method,
                "accuracy": "",
                "f1_macro": "",
                "time": "",
                "status": f"ERROR: {str(e)[:60]}",
            })

    # 生成柱状图
    if results_list:
        bar_data = [{"method": r["method"], "accuracy": r["accuracy"],
                     "f1_macro": r["f1_macro"]} for r in results_list]
        fig, _ = plot_classifier_comparison(bar_data, title_prefix="PHM 2010 ")
        fig.savefig(output_dir / "phm2010_ml_comparison.png", dpi=200)
        plt.close(fig)

        # 混淆矩阵
        fig, _ = plot_confusion_matrix_grid(results_list, title_prefix="PHM 2010 ")
        fig.savefig(output_dir / "phm2010_confusion_matrix.png", dpi=200)
        plt.close(fig)

    # 保存 CSV
    csv_path = output_dir / "summary.csv"
    keys = ["method", "accuracy", "f1_macro", "time", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report_rows)

    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
