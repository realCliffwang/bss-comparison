"""
PHM 2010 分类器对比实验
用法: python -m experiments.comparison.phm_classifiers
"""

from sklearn.model_selection import train_test_split

from bss_test.ml_classifier import train_classifier, evaluate_classifier
from bss_test.utils.logger import setup_logging

from experiments._common import (
    load_phm_features, run_classifier_comparison,
    save_comparison_plots, write_csv_summary, init_experiment,
)

CLASSIFIERS = ["svm", "rf", "knn", "lda"]

try:
    import xgboost
    CLASSIFIERS.insert(2, "xgb")
except ImportError:
    pass


def main():
    config, output_dir, logger = init_experiment(
        "PHM 2010 分类器对比实验", "classifier_comparison", "configs/phm2010.yaml",
    )
    logger.info(f"分类器: {[c.upper() for c in CLASSIFIERS]}")

    # 使用全部 6 个刀具: c1-c3 低磨损, c4 中磨损, c5-c6 高磨损
    tool_labels = {
        1: "low_wear", 2: "low_wear", 3: "low_wear",
        4: "medium_wear", 5: "high_wear", 6: "high_wear",
    }
    X, y = load_phm_features(config, tool_labels=tool_labels, seg_sec=0.2, overlap=0.5)

    if len(X) == 0:
        logger.error("无数据，退出")
        return

    logger.info(f"\n数据形状: {X.shape}, 类别: {set(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    logger.info(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    results = run_classifier_comparison(
        train_classifier, evaluate_classifier,
        X_train, y_train, X_test, y_test, CLASSIFIERS,
    )

    save_comparison_plots(results, "PHM2010", output_dir)

    csv_rows = [{
        "method": r["method"], "accuracy": f"{r['accuracy']:.4f}",
        "f1_macro": f"{r['f1_macro']:.4f}", "time": f"{r['time']:.2f}",
    } for r in results]
    write_csv_summary(output_dir, csv_rows)

    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
