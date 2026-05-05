"""
WDCNN vs BSS 对比实验
用法: python -m experiments.comparison.wdcnn_vs_bss

对比：
- WDCNN：端到端深度学习（需要训练数据和标签）
- BSS (SOBI)：无监督盲源分离 + 包络谱故障频率诊断
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from bss_test import ExperimentConfig
from bss_test.tfa import build_observation_matrix
from bss_test.bss import bss_factory
from bss_test.wdcnn import segment_signals, train_wdcnn, evaluate_wdcnn
from bss_test.evaluation import plot_confusion_matrix_grid
from bss_test.utils.logger import setup_logging

from experiments._common import (
    load_cwru_signals, load_cwru_raw_segments,
    write_csv_summary, init_experiment,
)


def bss_classify(signal, fs, fault_freqs, n_sources=5):
    """Use BSS + envelope spectrum for fault diagnosis.

    Returns
    -------
    str : predicted fault type
    dict : peaks at each fault frequency
    """
    cwt_config = {
        "mode": "single_channel_expansion",
        "tfa_method": "cwt",
        "n_bands": 20,
        "freq_range": (100, 5000),
        "wavelet": "cmor1.5-1.0",
    }
    X, _ = build_observation_matrix(signal.reshape(1, -1), fs, cwt_config)
    S_est, _, _ = bss_factory(X, method="sobi", n_components=n_sources)

    sig = S_est[0]
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    N = len(envelope)
    env_spec = np.abs(np.fft.rfft(envelope))
    freq = np.fft.rfftfreq(N, 1.0 / fs)

    fault_peaks = {}
    for name, fval in fault_freqs.items():
        mask = (freq >= fval - 5.0) & (freq <= fval + 5.0)
        fault_peaks[name] = float(np.max(env_spec[mask])) if np.sum(mask) > 0 else 0.0

    if not fault_peaks:
        return "unknown", fault_peaks

    predicted_freq_name = max(fault_peaks, key=fault_peaks.get)
    freq_to_fault = {"BPFO": "outer_race_6_007", "BPFI": "inner_race_007", "BSF": "ball_007"}
    return freq_to_fault.get(predicted_freq_name, "unknown"), fault_peaks


def main():
    config, output_dir, logger = init_experiment(
        "WDCNN vs BSS 对比实验", "wdcnn_vs_bss", "configs/cwru.yaml",
    )
    logger.info("WDCNN: 端到端深度学习（需要训练数据和标签）")
    logger.info("BSS: 盲源分离 + 包络谱诊断（不需要训练数据）")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    fault_types = ["normal", "inner_race_007", "ball_007", "outer_race_6_007"]
    fault_freqs = config.feature_freqs

    # ---- WDCNN ----
    logger.info("\n  === WDCNN ===")
    X_raw, y_raw = load_cwru_raw_segments(config, fault_types, segment_length=1024, overlap=0.5)
    if len(X_raw) == 0:
        logger.error("  No data")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw,
    )
    logger.info(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    start_time = time.time()
    model_dict = train_wdcnn(X_train, y_train, n_epochs=50, batch_size=32)
    wdcnn_time = time.time() - start_time
    wdcnn_metrics = evaluate_wdcnn(model_dict, X_test, y_test)
    logger.info(f"    Accuracy: {wdcnn_metrics['accuracy']:.4f}")
    logger.info(f"    F1-Macro: {wdcnn_metrics['f1_macro']:.4f}")
    logger.info(f"    Time: {wdcnn_time:.2f}s")

    # ---- BSS ----
    logger.info("\n  === BSS + 包络谱 ===")
    bss_predictions, bss_true_labels, bss_times = [], [], []

    for fault_type in fault_types:
        signals_pre, fs_pre = load_cwru_signals(config, fault_type, max_duration=2.0)
        sig = signals_pre[0]
        seg_len, step = 1024, 512
        n_segments = min((len(sig) - seg_len) // step + 1, 20)

        for seg_idx in range(n_segments):
            start = seg_idx * step
            seg = sig[start:start + seg_len]
            try:
                t0 = time.time()
                predicted, _ = bss_classify(seg, fs_pre, fault_freqs)
                bss_times.append(time.time() - t0)
                bss_predictions.append(predicted)
                bss_true_labels.append(fault_type)
            except Exception as e:
                logger.warning(f"    BSS failed ({fault_type} seg {seg_idx}): {e}")

    bss_results = []
    if bss_predictions:
        bss_accuracy = accuracy_score(bss_true_labels, bss_predictions)
        bss_f1 = f1_score(bss_true_labels, bss_predictions, average="macro", zero_division=0)
        bss_total_time = sum(bss_times)
        logger.info(f"    Accuracy: {bss_accuracy:.4f}")
        logger.info(f"    F1-Macro: {bss_f1:.4f}")
        logger.info(f"    Time: {bss_total_time:.2f}s")

        le = LabelEncoder()
        le.fit(fault_types + ["unknown"])
        y_true_enc = le.transform(bss_true_labels)
        y_pred_enc = le.transform(bss_predictions)
        bss_cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(le.classes_)))
        present_labels = sorted(set(bss_true_labels + bss_predictions))
        present_indices = [le.transform([l])[0] for l in present_labels]
        bss_cm_filtered = bss_cm[np.ix_(present_indices, present_indices)]

        bss_results = [{
            "method": "BSS (SOBI)", "accuracy": bss_accuracy, "f1_macro": bss_f1,
            "confusion_matrix": bss_cm_filtered, "label_names": present_labels,
            "time": bss_total_time,
        }]

    # ---- Summary ----
    results_list = [{
        "method": "WDCNN", "accuracy": wdcnn_metrics["accuracy"],
        "f1_macro": wdcnn_metrics["f1_macro"],
        "confusion_matrix": wdcnn_metrics["confusion_matrix"],
        "label_names": wdcnn_metrics["label_names"], "time": wdcnn_time,
    }] + bss_results

    if results_list:
        fig, _ = plot_confusion_matrix_grid(results_list, title_prefix="CWRU ")
        fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)

    csv_rows = [{"method": r["method"], "accuracy": f"{r['accuracy']:.4f}",
                 "f1_macro": f"{r['f1_macro']:.4f}", "time": f"{r['time']:.2f}"}
                for r in results_list]
    write_csv_summary(output_dir, csv_rows)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
