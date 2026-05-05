"""
ML 分类器对比实验
用法: python -m experiments.comparison.ml_classifiers
"""

from sklearn.model_selection import train_test_split

from bss_test.ml_classifier import train_classifier, evaluate_classifier
from bss_test.utils.logger import setup_logging

from experiments._common import (
    load_cwru_features, load_phm_features,
    run_classifier_comparison, save_comparison_plots,
    write_csv_summary, init_experiment,
)

CLASSIFIERS = ["svm", "rf", "knn", "lda"]

try:
    import xgboost
    CLASSIFIERS.insert(2, "xgb")
except ImportError:
    pass


def main():
    config, output_dir, logger = init_experiment(
        "ML 分类器对比实验", "ml_comparison", "configs/cwru.yaml",
    )
    logger.info(f"分类器: {[c.upper() for c in CLASSIFIERS]}")

    all_report_rows = []

    for dataset_name, config_path, loader in [
        ("CWRU", "configs/cwru.yaml", load_cwru_features),
        ("PHM2010", "configs/phm2010.yaml", load_phm_features),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"[{dataset_name}]")
        logger.info("=" * 60)

        ds_config, ds_output_dir, _ = init_experiment(
            f"{dataset_name} 分类", "ml_comparison", config_path,
        )
        X, y = loader(ds_config)

        if len(X) == 0:
            logger.warning(f"  {dataset_name} 无数据")
            continue

        logger.info(f"\n  数据形状: {X.shape}, 类别: {set(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        logger.info(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        results = run_classifier_comparison(
            train_classifier, evaluate_classifier,
            X_train, y_train, X_test, y_test, CLASSIFIERS,
        )

        save_comparison_plots(results, dataset_name, ds_output_dir)

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
