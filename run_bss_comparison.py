"""
BSS Method Comparison — Cross-Dataset Evaluation
=================================================

Compare SOBI, FastICA, JADE, PICARD on CWRU, PHM 2010, NASA Milling datasets.

Metrics:
  - Source Independence: mean |off-diagonal correlation| (lower = better)
  - FFDS: Fault Frequency Detection Score (higher = better)

Usage:
  python run_bss_comparison.py
"""

import sys
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_cwru, load_phm_cut, load_nasa_milling_single
from preprocessing import preprocess_signals
from cwt_module import build_observation_matrix, cwt_transform
from bss_module import bss_factory
from evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_correlation_matrix,
)


# ============================================================
# Configuration
# ============================================================

BSS_METHODS = ["sobi", "fastica", "jade", "picard"]

DATASETS = {
    "cwru": {
        "loader": "cwru",
        "fault_types": ["inner_race_007", "ball_007", "outer_race_6_007"],
        "load": 0,
        "channels": ["DE"],
        "preprocess": {"detrend": True, "bandpass": (100, 5000), "normalize": "zscore"},
        "cwt": {"wavelet": "cmor1.5-1.0", "n_bands": 8, "freq_range": (100, 5000), "mode": "single_channel_expansion"},
        "n_sources": 5,
        "feature_freqs": {"BPFO": 107.3, "BPFI": 162.2, "BSF": 70.6},
    },
    "phm2010": {
        "loader": "phm",
        "tool_ids": [1, 4],
        "rep_cut": 150,
        "sensor_types": ["vib_x", "vib_y", "vib_z"],
        "preprocess": {"detrend": True, "bandpass": (100, 20000), "normalize": "zscore"},
        "cwt": {"wavelet": "cmor1.5-1.0", "n_bands": 8, "freq_range": (100, 20000), "bands_per_ch": 3, "mode": "multi_channel"},
        "n_sources": 6,
        "feature_freqs": {"spindle": 173.3},  # 10400 RPM / 60
    },
    "nasa": {
        "loader": "nasa",
        "run_indices": [0, 10, 50],
        "sensor_types": ["vib_table", "vib_spindle", "force_ac"],
        "preprocess": {"detrend": True, "bandpass": None, "normalize": "zscore"},
        "cwt": {"wavelet": "cmor1.5-1.0", "n_bands": 8, "freq_range": (5, 100), "mode": "multi_channel"},
        "n_sources": 5,
        "feature_freqs": {"spindle": 166.7},  # ~10000 RPM
    },
}

OUTPUT_DIR = "outputs/bss_comparison"


# ============================================================
# Data Loading Functions
# ============================================================

def load_cwru_data(config):
    """Load CWRU dataset, return list of (name, signal_1d, fs, fault_freqs)."""
    results = []
    for ft in config["fault_types"]:
        try:
            signals, fs, rpm = load_cwru(
                data_dir="data/cwru", fault_type=ft,
                load=config["load"], channels=config["channels"],
            )
            # Use first channel, limit to 2 seconds
            sig = signals[0]
            n_use = min(len(sig), int(2.0 * fs))
            sig = sig[:n_use]
            results.append((f"cwru_{ft}", sig.reshape(1, -1), fs, config["feature_freqs"]))
            print(f"    [{ft}] loaded: {n_use} samples @ {fs} Hz")
        except Exception as e:
            print(f"    [{ft}] SKIPPED: {e}")
    return results


def load_phm_data(config):
    """Load PHM 2010 dataset, return list of (name, signals, fs, fault_freqs)."""
    results = []
    for tool_id in config["tool_ids"]:
        try:
            signals, fs, _ = load_phm_cut(
                tool_id=tool_id, cut_no=config["rep_cut"],
                data_dir="data/phm2010_milling",
                sensor_types=config["sensor_types"],
            )
            n_use = min(signals.shape[1], int(1.0 * fs))
            signals = signals[:, :n_use]
            results.append((f"phm_c{tool_id}", signals, fs, config["feature_freqs"]))
            print(f"    [c{tool_id}] loaded: {signals.shape} @ {fs} Hz")
        except Exception as e:
            print(f"    [c{tool_id}] SKIPPED: {e}")
    return results


def load_nasa_data(config):
    """Load NASA milling dataset, return list of (name, signals, fs, fault_freqs)."""
    results = []
    for idx in config["run_indices"]:
        try:
            signals, meta, fs = load_nasa_milling_single(
                run_index=idx, data_dir="data/phm2010_milling",
                sensor_types=config["sensor_types"],
            )
            n_use = min(signals.shape[1], int(5.0 * fs))
            signals = signals[:, :n_use]
            name = f"nasa_run{idx}_c{meta['case']}"
            results.append((name, signals, fs, config["feature_freqs"]))
            print(f"    [run {idx}] loaded: {signals.shape} @ {fs} Hz (case={meta['case']})")
        except Exception as e:
            print(f"    [run {idx}] SKIPPED: {e}")
    return results


LOADERS = {
    "cwru": load_cwru_data,
    "phm2010": load_phm_data,
    "nasa": load_nasa_data,
}


# ============================================================
# BSS Comparison Core
# ============================================================

def preprocess_signal(signals, fs, config):
    """Preprocess and limit duration."""
    signals_pre, fs_pre = preprocess_signals(signals, fs, config)
    return signals_pre, fs_pre


def run_single_bss(signal_1d, fs, tfa_config, bss_method, n_sources):
    """Run a single BSS method on a 1D signal."""
    # Build observation matrix via CWT
    matrix, freqs = build_observation_matrix(
        signal_1d.reshape(1, -1), fs, tfa_config
    )
    # Run BSS
    S_est, A_est, W = bss_factory(matrix, method=bss_method, n_components=n_sources)
    return S_est, matrix


def compare_bss_methods(signals, fs, fault_freqs, tfa_config, n_sources, methods, out_dir, sample_name):
    """Compare all BSS methods on a single signal."""
    results = {}

    for method in methods:
        try:
            S_est, X_obs = run_single_bss(signals[0], fs, tfa_config, method, n_sources)

            # Compute metrics
            indep = compute_independence_metric(S_est)
            ffds = compute_fault_detection_score(S_est, fs, fault_freqs)

            results[method] = {
                "S_est": S_est,
                "X_obs": X_obs,
                "independence": indep,
                "ffds": ffds,
                "status": "OK",
            }
        except Exception as e:
            results[method] = {
                "S_est": None,
                "independence": float("nan"),
                "ffds": float("nan"),
                "status": f"ERROR: {str(e)[:60]}",
            }

    return results


# ============================================================
# Visualization
# ============================================================

def plot_bss_comparison_grid(results, fs, fault_freqs, sample_name, out_dir):
    """Plot envelope spectrum comparison for all BSS methods."""
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 4 * n_methods))
    if n_methods == 1:
        axes = [axes]

    fig.suptitle(f"BSS Method Comparison — {sample_name}", fontsize=14, fontweight="bold")

    for ax, (method, res) in zip(axes, results.items()):
        if res["S_est"] is None:
            ax.text(0.5, 0.5, f"{method.upper()}\n{res['status']}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{method.upper()} — FAILED")
            continue

        S_est = res["S_est"]
        sig = S_est[0]  # Use first separated source

        # Compute envelope spectrum
        from scipy.signal import hilbert
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        N = len(envelope)
        env_spec = np.abs(np.fft.rfft(envelope))
        freq = np.fft.rfftfreq(N, 1.0 / fs)

        # Plot
        ax.plot(freq, env_spec, linewidth=0.5, color="steelblue")
        ax.set_xlim(0, min(500, fs / 2))
        ax.set_ylabel("Amplitude")

        # Mark fault frequencies
        colors = ["red", "green", "orange", "purple"]
        for i, (name, fval) in enumerate(fault_freqs.items()):
            if fval < fs / 2:
                ax.axvline(fval, color=colors[i % len(colors)], linestyle="--",
                           alpha=0.7, linewidth=1.5, label=f"{name}={fval:.1f}Hz")

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{method.upper()} — Independence={res['independence']:.3f}, FFDS={res['ffds']:.1f}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    return fig


def plot_metrics_bar_chart(all_results, out_dir):
    """Plot bar chart comparing FFDS across datasets and methods."""
    datasets = sorted(all_results.keys())
    methods = BSS_METHODS

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("BSS Method Comparison Across Datasets", fontsize=14, fontweight="bold")

    # FFDS comparison
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.18
    for i, method in enumerate(methods):
        ffds_vals = []
        for ds in datasets:
            ds_results = all_results[ds]
            # Average FFDS across samples in this dataset
            vals = [r.get(method, {}).get("ffds", 0) for r in ds_results.values()]
            vals = [v for v in vals if not np.isnan(v)]
            ffds_vals.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, ffds_vals, width, label=method.upper())

    ax.set_xlabel("Dataset")
    ax.set_ylabel("FFDS (higher = better)")
    ax.set_title("Fault Frequency Detection Score")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Independence comparison
    ax = axes[1]
    for i, method in enumerate(methods):
        indep_vals = []
        for ds in datasets:
            ds_results = all_results[ds]
            vals = [r.get(method, {}).get("independence", 1) for r in ds_results.values()]
            vals = [v for v in vals if not np.isnan(v)]
            indep_vals.append(np.mean(vals) if vals else 1)
        ax.bar(x + i * width, indep_vals, width, label=method.upper())

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Independence (lower = better)")
    ax.set_title("Source Independence")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_summary_heatmap(all_results, out_dir):
    """Plot heatmap: datasets x methods, color = FFDS."""
    datasets = sorted(all_results.keys())
    methods = BSS_METHODS

    # Build matrix
    n_ds = len(datasets)
    n_m = len(methods)
    ffds_matrix = np.zeros((n_ds, n_m))
    indep_matrix = np.zeros((n_ds, n_m))

    for i, ds in enumerate(datasets):
        ds_results = all_results[ds]
        for j, method in enumerate(methods):
            vals_ffds = [r.get(method, {}).get("ffds", 0) for r in ds_results.values()]
            vals_indep = [r.get(method, {}).get("independence", 1) for r in ds_results.values()]
            vals_ffds = [v for v in vals_ffds if not np.isnan(v)]
            vals_indep = [v for v in vals_indep if not np.isnan(v)]
            ffds_matrix[i, j] = np.mean(vals_ffds) if vals_ffds else 0
            indep_matrix[i, j] = np.mean(vals_indep) if vals_indep else 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("BSS Performance Heatmap", fontsize=14, fontweight="bold")

    # FFDS heatmap
    ax = axes[0]
    im = ax.imshow(ffds_matrix, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(datasets)
    ax.set_title("FFDS (higher = better)")
    for i in range(n_ds):
        for j in range(n_m):
            ax.text(j, i, f"{ffds_matrix[i, j]:.1f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax)

    # Independence heatmap
    ax = axes[1]
    im = ax.imshow(indep_matrix, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(datasets)
    ax.set_title("Independence (lower = better)")
    for i in range(n_ds):
        for j in range(n_m):
            ax.text(j, i, f"{indep_matrix[i, j]:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig, ffds_matrix, indep_matrix


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_rows = []
    all_results = {}  # dataset -> {sample_name -> {method -> metrics}}

    print("=" * 60)
    print("BSS Method Cross-Dataset Comparison")
    print("=" * 60)
    print(f"Methods: {[m.upper() for m in BSS_METHODS]}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print()

    for ds_name, ds_config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        ds_out = os.path.join(OUTPUT_DIR, ds_name)
        os.makedirs(ds_out, exist_ok=True)

        # Load data
        print(f"  Loading data...")
        loader = LOADERS[ds_name]
        samples = loader(ds_config)

        if not samples:
            print(f"  ERROR: No data loaded for {ds_name}. Skipping.")
            continue

        all_results[ds_name] = {}

        for sample_name, signals, fs, fault_freqs in samples:
            print(f"\n  --- {sample_name} ---")
            print(f"  Signals: {signals.shape} @ {fs} Hz")

            # Preprocess
            signals_pre, fs_pre = preprocess_signal(signals, fs, ds_config["preprocess"])
            print(f"  Preprocessed: {signals_pre.shape} @ {fs_pre} Hz")

            # Run BSS comparison
            tfa_config = ds_config["cwt"]
            n_sources = ds_config["n_sources"]

            print(f"  Running BSS methods...")
            results = compare_bss_methods(
                signals_pre, fs_pre, fault_freqs,
                tfa_config, n_sources, BSS_METHODS,
                ds_out, sample_name,
            )

            all_results[ds_name][sample_name] = results

            # Print results
            print(f"\n  Results for {sample_name}:")
            print(f"  {'Method':<10} {'Independence':<15} {'FFDS':<10} {'Status'}")
            print(f"  {'-'*50}")
            for method, res in results.items():
                print(f"  {method.upper():<10} {res['independence']:<15.3f} {res['ffds']:<10.1f} {res['status']}")

                report_rows.append({
                    "dataset": ds_name,
                    "sample": sample_name,
                    "method": method,
                    "independence": f"{res['independence']:.4f}",
                    "ffds": f"{res['ffds']:.2f}",
                    "status": res["status"],
                })

            # Plot envelope spectrum comparison
            fig = plot_bss_comparison_grid(results, fs_pre, fault_freqs, sample_name, ds_out)
            fig.savefig(os.path.join(ds_out, f"{sample_name}_bss_comparison.png"), dpi=150)
            plt.close(fig)
            print(f"  Saved: {ds_out}/{sample_name}_bss_comparison.png")

    # ============================================================
    # Cross-dataset summary
    # ============================================================
    print(f"\n{'='*60}")
    print("Generating Cross-Dataset Summary")
    print(f"{'='*60}")

    if all_results:
        # Bar chart
        fig = plot_metrics_bar_chart(all_results, OUTPUT_DIR)
        fig.savefig(os.path.join(OUTPUT_DIR, "cross_dataset_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR}/cross_dataset_comparison.png")

        # Heatmap
        fig, ffds_mat, indep_mat = plot_summary_heatmap(all_results, OUTPUT_DIR)
        fig.savefig(os.path.join(OUTPUT_DIR, "performance_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR}/performance_heatmap.png")

        # Find best method per dataset
        print(f"\n  Best Method per Dataset:")
        datasets = sorted(all_results.keys())
        for i, ds in enumerate(datasets):
            best_method = max(BSS_METHODS, key=lambda m: ffds_mat[i, BSS_METHODS.index(m)])
            print(f"    {ds}: {best_method.upper()} (FFDS={ffds_mat[i, BSS_METHODS.index(best_method)]:.1f})")

    # Save CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    keys = ["dataset", "sample", "method", "independence", "ffds", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"\n  Summary CSV: {csv_path}")

    print(f"\n{'='*60}")
    print(f"ALL DONE — results in {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
