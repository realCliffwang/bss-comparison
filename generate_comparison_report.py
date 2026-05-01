"""
Comprehensive comparison report — TFA x BSS cross-comparison + ML methods.

PIPELINE:
  1. TFA x BSS cross-comparison: each (tfa, bss) pair -> separation -> envelope spectrum
  2. ML comparison: extract features -> all classifiers -> bar chart + confusion grid
  3. Summary CSV -> outputs/comparison/results_summary.csv

Usage:
  python generate_comparison_report.py
"""

import sys
import os
import numpy as np
import csv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import load_cwru
from preprocessing import preprocess_signals
from cwt_module import time_freq_factory
from bss_module import bss_factory
from evaluation import (
    plot_tfa_bss_cross_comparison,
    plot_separation_quality_report,
    plot_classifier_comparison,
    plot_confusion_matrix_grid,
)
from feature_extractor import extract_features
from ml_classifier import train_classifier, evaluate_classifier


COMPARISON_CONFIG = {
    # TFA methods to include in cross-comparison
    "tfa_methods": ["cwt", "stft", "emd", "eemd"],
    # BSS methods to include in cross-comparison
    "bss_methods": ["sobi", "fastica", "jade", "picard"],
    # ML classifiers to compare
    "ml_methods": ["svm", "rf", "knn", "lda"],
    # Dataset: "cwru"
    "dataset": "cwru",
    "data_dir": "data/cwru",
    "fault_types": ["inner_race_007", "ball_007", "outer_race_6_007"],
    "load": 0,
    "channels": ["DE"],
    # Preprocessing
    "preprocess": {
        "detrend": True,
        "bandpass": (100, 5000),
        "normalize": "zscore",
    },
    # TFA shared params
    "n_bands": 8,
    "freq_range": (100, 5000),
    "wavelet": "cmor1.5-1.0",
    # BSS params
    "n_sources": 5,
    # CWRU bearing characteristic frequencies (6205 bearing @ ~1797 RPM)
    "feature_freqs": {
        "BPFO": 107.3,
        "BPFI": 162.2,
        "BSF": 70.6,
    },
    # Output
    "output_dir": "outputs/comparison",
}


def preprocess_signal(signals, fs, config):
    signals_pre, fs_pre = preprocess_signals(signals, fs, config)
    n_use = min(signals_pre.shape[1], int(2.0 * fs_pre))
    return signals_pre[:, :n_use], fs_pre


def _build_obs_matrix(signal, fs, tfa_method):
    """Build observation matrix using a specific TFA method."""
    matrix, freqs = time_freq_factory(
        signal, fs, method=tfa_method,
        n_bands=COMPARISON_CONFIG["n_bands"],
        freq_range=COMPARISON_CONFIG["freq_range"],
        wavelet=COMPARISON_CONFIG["wavelet"],
    )
    labels = [f"ch0_f{freqs[i]:.0f}Hz" for i in range(matrix.shape[0])]
    return matrix, labels


def run_tfa_bss_cross_comparison(signal, fs, tfa_methods, bss_methods,
                                  out_dir, report_rows):
    """Step 1+2 merged: cross-compare every TFA x BSS pair."""
    print("\n" + "=" * 60)
    print("STEP 1: TFA x BSS Cross-Comparison")
    print("=" * 60)

    n_tfa = len(tfa_methods)
    n_bss = len(bss_methods)
    print(f"  Grid: {n_tfa} TFA x {n_bss} BSS = {n_tfa * n_bss} combinations")

    cross_results = {}

    for tfa in tfa_methods:
        print(f"\n  --- TFA: {tfa.upper()} ---")

        try:
            X_for_bss, obs_labels = _build_obs_matrix(signal, fs, tfa)
            n_obs = X_for_bss.shape[0]
            print(f"    Observation matrix: {n_obs} obs x {X_for_bss.shape[1]} samples")
        except Exception as e:
            print(f"    FAILED to build obs matrix: {e}")
            for bss in bss_methods:
                report_rows.append({
                    "tfa_method": tfa, "bss_method": bss,
                    "n_sources": 0, "corr_mean": "", "ffds": "",
                    "status": f"OBS_ERR: {e}",
                })
            continue

        for bss in bss_methods:
            combo = f"{tfa}+{bss}"
            print(f"    [{bss}] ", end="", flush=True)
            try:
                S_est, A_est, W = bss_factory(
                    X_for_bss, method=bss,
                    n_components=COMPARISON_CONFIG["n_sources"],
                )
                cross_results[(tfa, bss)] = S_est

                # Compute quality metrics
                from evaluation import (compute_independence_metric,
                                        compute_fault_detection_score)
                indep = compute_independence_metric(S_est)
                ffds = compute_fault_detection_score(
                    S_est, fs, COMPARISON_CONFIG["feature_freqs"])

                report_rows.append({
                    "tfa_method": tfa, "bss_method": bss,
                    "n_sources": S_est.shape[0],
                    "corr_mean": f"{indep:.4f}",
                    "ffds": f"{ffds:.2f}",
                    "status": "OK",
                })
                print(f"OK ({S_est.shape[0]} src, ind={indep:.3f}, FFDS={ffds:.1f})")
            except Exception as e:
                print(f"SKIPPED ({str(e)[:60]})")
                report_rows.append({
                    "tfa_method": tfa, "bss_method": bss,
                    "n_sources": 0, "corr_mean": "", "ffds": "",
                    "status": f"ERROR: {str(e)[:80]}",
                })

    # Generate cross-comparison grid figure
    if cross_results:
        print(f"\n  Generating TFA x BSS cross-comparison grid "
              f"({len(cross_results)} cells)...")

        fig, _ = plot_tfa_bss_cross_comparison(
            cross_results, fs,
            fault_freqs=COMPARISON_CONFIG["feature_freqs"],
            title_prefix="CWRU Inner Race Fault — "
        )
        path = os.path.join(out_dir, "tfa_bss_cross_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

        fig, _ = plot_tfa_bss_cross_comparison(
            cross_results, fs,
            fault_freqs=None,
            title_prefix="CWRU Inner Race Fault — "
        )
        path = os.path.join(out_dir, "tfa_bss_cross_comparison_clean.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

        # Quality report (envelope grid + dual heatmaps + BEST highlight)
        print(f"\n  Generating separation quality report...")
        fig, metrics = plot_separation_quality_report(
            cross_results, fs,
            fault_freqs=COMPARISON_CONFIG["feature_freqs"],
            title_prefix="CWRU Inner Race Fault — ",
        )
        path = os.path.join(out_dir, "separation_quality_report.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  Saved: {path}")

        # Print best combination
        valid = [(k, v) for k, v in metrics.items()
                 if not np.isnan(v["ffds"]) and v["ffds"] > 0]
        if valid:
            best = max(valid, key=lambda x: (x[1]["ffds"], -x[1]["independence"]))
            print(f"\n  ★ BEST: {best[0][0].upper()} + {best[0][1].upper()}")
            print(f"      Independence={best[1]['independence']:.3f}, "
                  f"FFDS={best[1]['ffds']:.1f}")

    return cross_results


def run_ml_comparison(fault_type_signals, fs, methods, out_dir, report_rows):
    """Step 3: Compare ML classifiers on fault classification."""
    print("\n" + "=" * 60)
    print("STEP 2: ML Classifier Comparison")
    print("=" * 60)

    X_all = []
    y_all = []
    for label, signals_list in fault_type_signals.items():
        for sig_ch in signals_list:
            feat = extract_features(sig_ch, fs, feature_set="all")
            X_all.append(feat)
            y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    if len(np.unique(y_all)) < 2 or len(X_all) < 4:
        print("  Not enough data for ML comparison. Skipping.")
        return

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    results_list = []
    for method in methods:
        print(f"  [{method}] ", end="", flush=True)
        try:
            model = train_classifier(X_tr, y_tr, method=method)
            metrics = evaluate_classifier(model, X_te, y_te)
            results_list.append({
                "method": method,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "confusion_matrix": metrics["confusion_matrix"],
                "label_names": metrics["label_names"],
            })
            report_rows.append({
                "tfa_method": "ml", "bss_method": method,
                "n_sources": 0, "corr_mean": "", "ffds": "",
                "status": f"acc={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f}",
            })
            print(f"OK (acc={metrics['accuracy']:.3f}, f1={metrics['f1_macro']:.3f})")
        except Exception as e:
            print(f"SKIPPED ({str(e)[:60]})")
            report_rows.append({
                "tfa_method": "ml", "bss_method": method,
                "n_sources": 0, "corr_mean": "", "ffds": "",
                "status": f"ERROR: {str(e)[:80]}",
            })

    if results_list:
        fig, _ = plot_classifier_comparison(results_list,
                                             title_prefix="ML Comparison — ")
        path = os.path.join(out_dir, "ml_classifier_bar.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved: {path}")

        fig, _ = plot_confusion_matrix_grid(results_list,
                                             title_prefix="ML Comparison — ")
        path = os.path.join(out_dir, "ml_confusion_grid.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    return results_list


def save_summary_csv(report_rows, out_dir):
    """Save comparison summary to CSV."""
    path = os.path.join(out_dir, "results_summary.csv")
    keys = ["tfa_method", "bss_method", "n_sources", "corr_mean", "ffds", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"\nSummary saved: {path}")


def main():
    config = COMPARISON_CONFIG
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("BSS-TEST: TFA x BSS Cross-Comparison Report")
    print("=" * 60)

    report_rows = []

    # ================================================================
    # Data loading
    # ================================================================
    print("\n[0] Loading CWRU data...")
    data_dir = config["data_dir"]
    fault_types = config["fault_types"]
    load_val = config["load"]
    channels = config["channels"]

    fault_signals = {}
    for ft in fault_types:
        try:
            signals, fs, rpm = load_cwru(
                data_dir=data_dir, fault_type=ft, load=load_val,
                channels=channels,
            )
            fault_signals[ft] = signals[0]
            print(f"  [{ft}] loaded: {signals.shape[1]} samples @ {fs} Hz")
        except Exception as e:
            print(f"  [{ft}] SKIPPED: {e}")

    if not fault_signals:
        print("ERROR: No data loaded.")
        return

    # Use inner_race_007 as representative (has strongest fault signature)
    rep_ft = fault_types[0]
    signal_raw = fault_signals[rep_ft]
    print(f"\n  Representative signal: [{rep_ft}]")

    # Preprocess
    signals_2d = signal_raw.reshape(1, -1)
    signal_pre, fs_pre = preprocess_signal(signals_2d, fs, config["preprocess"])
    signal_1d = signal_pre[0]
    print(f"  Preprocessed: {signal_1d.shape[0]} samples @ {fs_pre} Hz")

    # ================================================================
    # Step 1: TFA x BSS Cross-Comparison
    # ================================================================
    run_tfa_bss_cross_comparison(
        signal_1d, fs_pre,
        config["tfa_methods"], config["bss_methods"],
        out_dir, report_rows,
    )

    # ================================================================
    # Step 2: ML Comparison
    # ================================================================
    ml_data = {}
    for ft in fault_types:
        if ft in fault_signals:
            sig = fault_signals[ft]
            n_total = len(sig)
            n_seg = min(6, n_total // 5000)
            seg_len = n_total // max(n_seg, 1)
            segments = [sig[s * seg_len:(s + 1) * seg_len] for s in range(n_seg)]
            if segments:
                ml_data[ft] = segments

    if len(ml_data) >= 2:
        run_ml_comparison(ml_data, fs, config["ml_methods"],
                          out_dir, report_rows)

    # ================================================================
    # Save summary
    # ================================================================
    save_summary_csv(report_rows, out_dir)

    print("\n" + "=" * 60)
    print(f"ALL DONE — results in {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
