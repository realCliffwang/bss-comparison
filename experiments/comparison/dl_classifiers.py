"""
深度学习分类器对比实验
用法: python -m experiments.comparison.dl_classifiers
"""

import time

from sklearn.model_selection import train_test_split

from bss_test.dl_classifier import train_dl_classifier, evaluate_dl_classifier
from bss_test.utils.logger import setup_logging, get_logger

from experiments._common import (
    load_cwru_features, load_phm_features,
    save_comparison_plots, write_csv_summary, init_experiment,
)

logger = get_logger(__name__)

DL_METHODS = ["cnn", "lstm", "transformer"]


def run_dl_comparison(X, y, dataset_name, output_dir, n_epochs=50):
    """Run DL classifiers on a dataset."""
    from bss_test.evaluation import setup_academic_style
    setup_academic_style()

    logger.info(f"\n  数据形状: {X.shape}, 类别: {set(y)}")

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
            logger.info(f"      Time: {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"      Failed: {e}")

    save_comparison_plots(results_list, dataset_name, output_dir)
    return results_list


def main():
    config, output_dir, logger = init_experiment(
        "深度学习分类器对比实验", "dl_comparison", "configs/cwru.yaml",
    )
    logger.info(f"方法: {[m.upper() for m in DL_METHODS]}")

    all_report_rows = []

    for dataset_name, config_path, loader in [
        ("CWRU", "configs/cwru.yaml", load_cwru_features),
        ("PHM2010", "configs/phm2010.yaml", load_phm_features),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"[{dataset_name}]")
        logger.info("=" * 60)

        ds_config, ds_output_dir, _ = init_experiment(
            f"{dataset_name} DL 分类", "dl_comparison", config_path,
        )
        X, y = loader(ds_config)

        if len(X) == 0:
            logger.warning(f"  {dataset_name} 无数据")
            continue

        results = run_dl_comparison(X, y, dataset_name, ds_output_dir)

        for r in results:
            all_report_rows.append({
                "dataset": dataset_name,
                "method": r["method"],
                "accuracy": f"{r['accuracy']:.4f}",
                "f1_macro": f"{r['f1_macro']:.4f}",
                "time": f"{r['time']:.2f}",
            })

    write_csv_summary(output_dir, all_report_rows)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
