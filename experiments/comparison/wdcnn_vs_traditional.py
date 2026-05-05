"""
WDCNN vs 传统方法对比实验
用法: python -m experiments.comparison.wdcnn_vs_traditional
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test.ml_classifier import train_classifier, evaluate_classifier
from bss_test.dl_classifier import train_dl_classifier, evaluate_dl_classifier
from bss_test.wdcnn import train_wdcnn, evaluate_wdcnn
from bss_test.evaluation import (
    plot_classifier_comparison,
    plot_confusion_matrix_grid,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

from experiments._common import (
    load_cwru_features, load_phm_features,
    load_cwru_raw_segments, load_phm_raw_segments,
    run_classifier_comparison, save_comparison_plots,
    write_csv_summary, init_experiment,
)

ML_METHODS = ["svm", "rf", "knn", "lda"]
DL_METHODS = ["cnn", "lstm", "transformer"]

try:
    import xgboost
    ML_METHODS.insert(2, "xgb")
except ImportError:
    pass


def run_comparison(X_features, y_features, X_raw, y_raw, dataset_name, output_dir, n_epochs=50):
    """Run WDCNN vs traditional methods on a dataset."""
    from sklearn.model_selection import train_test_split

    logger.info(f"\n  特征: {X_features.shape}, 原始信号: {X_raw.shape}, 类别: {set(y_features)}")

    X_feat_train, X_feat_test, y_feat_train, y_feat_test = train_test_split(
        X_features, y_features, test_size=0.3, random_state=42, stratify=y_features,
    )
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw,
    )

    results_list = []

    # ML
    logger.info("\n  === ML ===")
    ml_results = run_classifier_comparison(
        train_classifier, evaluate_classifier,
        X_feat_train, y_feat_train, X_feat_test, y_feat_test, ML_METHODS,
    )
    results_list.extend(ml_results)

    # DL (feature input)
    logger.info("\n  === DL (features) ===")
    for dl_method in DL_METHODS:
        logger.info(f"\n    --- {dl_method.upper()} ---")
        try:
            start_time = time.time()
            model_dict = train_dl_classifier(
                X_feat_train, y_feat_train, method=dl_method,
                n_epochs=n_epochs, batch_size=32, learning_rate=0.001,
            )
            elapsed = time.time() - start_time
            metrics = evaluate_dl_classifier(model_dict, X_feat_test, y_feat_test)

            results_list.append({
                "method": dl_method, "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "confusion_matrix": metrics["confusion_matrix"],
                "label_names": metrics["label_names"], "time": elapsed,
            })
            logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"      F1-Macro: {metrics['f1_macro']:.4f}")
        except Exception as e:
            logger.error(f"      Failed: {e}")

    # WDCNN (raw signal)
    logger.info("\n  === WDCNN (raw) ===")
    try:
        start_time = time.time()
        model_dict = train_wdcnn(X_raw_train, y_raw_train, n_epochs=n_epochs, batch_size=32)
        elapsed = time.time() - start_time
        metrics = evaluate_wdcnn(model_dict, X_raw_test, y_raw_test)

        results_list.append({
            "method": "wdcnn", "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "confusion_matrix": metrics["confusion_matrix"],
            "label_names": metrics["label_names"], "time": elapsed,
        })
        logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"      F1-Macro: {metrics['f1_macro']:.4f}")
    except Exception as e:
        logger.error(f"      Failed: {e}")

    save_comparison_plots(results_list, dataset_name, output_dir)
    return results_list


def main():
    config, output_dir, logger = init_experiment(
        "WDCNN vs 传统方法对比实验", "wdcnn_comparison", "configs/cwru.yaml",
    )
    logger.info(f"ML: {[m.upper() for m in ML_METHODS]}")
    logger.info(f"DL: {[m.upper() for m in DL_METHODS]}")
    logger.info("WDCNN: raw signal end-to-end")

    all_report_rows = []

    for dataset_name, config_path in [("CWRU", "configs/cwru.yaml"), ("PHM2010", "configs/phm2010.yaml")]:
        logger.info(f"\n{'='*60}")
        logger.info(f"[{dataset_name}]")
        logger.info("=" * 60)

        ds_config, ds_output_dir, _ = init_experiment(
            f"{dataset_name} WDCNN对比", "wdcnn_comparison", config_path,
        )

        X_feat, y_feat = (load_cwru_features if "cwru" in config_path else load_phm_features)(ds_config)
        X_raw, y_raw = (load_cwru_raw_segments if "cwru" in config_path else load_phm_raw_segments)(ds_config)

        if len(X_feat) > 0 and len(X_raw) > 0:
            results = run_comparison(X_feat, y_feat, X_raw, y_raw, dataset_name, ds_output_dir)
            for r in results:
                all_report_rows.append({
                    "dataset": dataset_name, "method": r["method"],
                    "accuracy": f"{r['accuracy']:.4f}", "f1_macro": f"{r['f1_macro']:.4f}",
                    "time": f"{r['time']:.2f}",
                })

    write_csv_summary(output_dir, all_report_rows)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
