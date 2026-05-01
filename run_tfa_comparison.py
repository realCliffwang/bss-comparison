"""
TFA Method Comparison — Two-Phase Evaluation
=============================================

Phase 1: TFA特性评估（频谱分辨率、计算时间、故障频率能量比）
Phase 2: TFA × BSS 交叉对比（6种TFA × 4种BSS = 24种组合）

Supported TFA methods: CWT, STFT, WPT, EMD, EEMD, CEEMDAN
Supported BSS methods: SOBI, FastICA, JADE, PICARD

Usage:
  python run_tfa_comparison.py
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
from cwt_module import time_freq_factory, build_observation_matrix
from bss_module import bss_factory
from evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
)


# ============================================================
# Configuration
# ============================================================

TFA_METHODS = ["cwt", "stft", "wpt", "emd"]
BSS_METHODS = ["sobi", "fastica", "jade", "picard"]

DATASETS = {
    "cwru": {
        "loader": "cwru",
        "fault_types": ["inner_race_007", "ball_007", "outer_race_6_007"],
        "load": 0,
        "channels": ["DE"],
        "preprocess": {"detrend": True, "bandpass": (100, 5000), "normalize": "zscore"},
        "tfa_params": {"n_bands": 8, "freq_range": (100, 5000), "wavelet": "cmor1.5-1.0"},
        "n_sources": 5,
        "feature_freqs": {"BPFO": 107.3, "BPFI": 162.2, "BSF": 70.6},
    },
    "phm2010": {
        "loader": "phm",
        "tool_ids": [1, 4],
        "rep_cut": 150,
        "sensor_types": ["vib_x", "vib_y", "vib_z"],
        "preprocess": {"detrend": True, "bandpass": (100, 20000), "normalize": "zscore"},
        "tfa_params": {"n_bands": 8, "freq_range": (100, 20000), "wavelet": "cmor1.5-1.0"},
        "n_sources": 6,
        "feature_freqs": {"spindle": 173.3},
    },
    "nasa": {
        "loader": "nasa",
        "run_indices": [0, 10, 50],
        "sensor_types": ["vib_table", "vib_spindle", "force_ac"],
        "preprocess": {"detrend": True, "bandpass": None, "normalize": "zscore"},
        "tfa_params": {"n_bands": 8, "freq_range": (5, 100), "wavelet": "cmor1.5-1.0"},
        "n_sources": 5,
        "feature_freqs": {"spindle": 166.7},
    },
}

OUTPUT_DIR = "outputs/tfa_comparison"


# ============================================================
# Data Loading Functions
# ============================================================

def load_cwru_data(config):
    results = []
    for ft in config["fault_types"]:
        try:
            signals, fs, rpm = load_cwru(
                data_dir="data/cwru", fault_type=ft,
                load=config["load"], channels=config["channels"],
            )
            sig = signals[0]
            n_use = min(len(sig), int(2.0 * fs))
            sig = sig[:n_use]
            results.append((f"cwru_{ft}", sig.reshape(1, -1), fs, config["feature_freqs"]))
            print(f"    [{ft}] loaded: {n_use} samples @ {fs} Hz")
        except Exception as e:
            print(f"    [{ft}] SKIPPED: {e}")
    return results


def load_phm_data(config):
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
            print(f"    [run {idx}] loaded: {signals.shape} @ {fs} Hz")
        except Exception as e:
            print(f"    [run {idx}] SKIPPED: {e}")
    return results


LOADERS = {
    "cwru": load_cwru_data,
    "phm2010": load_phm_data,
    "nasa": load_nasa_data,
}


# ============================================================
# Phase 1: TFA Characteristic Evaluation
# ============================================================

def evaluate_tfa_characteristics(signal_1d, fs, tfa_method, tfa_params, fault_freqs):
    """Evaluate TFA method characteristics: time, resolution, energy concentration."""
    start_time = time.time()

    try:
        # WPT needs different wavelet (db4 instead of cmor)
        params = tfa_params.copy()
        if tfa_method == "wpt":
            params["wavelet"] = "db4"

        matrix, freqs = time_freq_factory(signal_1d, fs, method=tfa_method, **params)
        elapsed = time.time() - start_time

        n_bands = matrix.shape[0]
        freq_resolution = (freqs[-1] - freqs[0]) / n_bands if n_bands > 1 else 0

        # Energy concentration: entropy of normalized energy distribution
        band_energy = np.mean(matrix**2, axis=1)
        total_energy = np.sum(band_energy) + 1e-12
        energy_norm = band_energy / total_energy
        entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-12))

        # Fault frequency energy ratio
        fault_energy_ratio = 0
        for ff_name, ff_val in fault_freqs.items():
            idx = np.argmin(np.abs(freqs - ff_val))
            fault_energy_ratio += band_energy[idx]
        fault_energy_ratio /= total_energy

        return {
            "status": "OK",
            "elapsed": elapsed,
            "n_bands": n_bands,
            "freq_resolution": freq_resolution,
            "entropy": entropy,
            "fault_energy_ratio": fault_energy_ratio,
            "matrix": matrix,
            "freqs": freqs,
        }
    except Exception as e:
        return {
            "status": f"ERROR: {str(e)[:60]}",
            "elapsed": time.time() - start_time,
            "n_bands": 0,
            "freq_resolution": 0,
            "entropy": 0,
            "fault_energy_ratio": 0,
            "matrix": None,
            "freqs": None,
        }


def plot_tfa_spectrograms(results_by_method, fs, sample_name, out_dir):
    """Plot spectrograms for all TFA methods side by side."""
    valid = {k: v for k, v in results_by_method.items() if v["matrix"] is not None}
    if not valid:
        return

    n_methods = len(valid)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3 * n_methods))
    if n_methods == 1:
        axes = [axes]

    fig.suptitle(f"TFA Spectrograms — {sample_name}", fontsize=14, fontweight="bold")

    for ax, (method, res) in zip(axes, valid.items()):
        matrix = res["matrix"]
        freqs = res["freqs"]
        n_samples = matrix.shape[1]

        time_axis = np.arange(n_samples) / fs
        extent = [time_axis[0], time_axis[-1], freqs[0], freqs[-1]]

        coef_db = 20 * np.log10(matrix / (matrix.max(axis=1, keepdims=True) + 1e-12) + 1e-12)

        im = ax.imshow(coef_db, aspect="auto", origin="lower", extent=extent, cmap="jet")
        ax.set_ylabel("Freq [Hz]")
        ax.set_title(f"{method.upper()} — {res['n_bands']} bands, "
                     f"time={res['elapsed']:.2f}s, entropy={res['entropy']:.2f}")
        fig.colorbar(im, ax=ax, label="Magnitude [dB]")

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()

    path = os.path.join(out_dir, f"{sample_name}_tfa_spectrograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# Phase 2: TFA × BSS Cross-Comparison
# ============================================================

def run_cross_comparison(signal_1d, fs, tfa_methods, bss_methods, tfa_params,
                         n_sources, fault_freqs, out_dir, sample_name):
    """Run TFA × BSS cross-comparison."""
    results = {}

    for tfa in tfa_methods:
        print(f"    TFA: {tfa.upper()}", end="", flush=True)

        # Build observation matrix
        try:
            # WPT needs different wavelet (db4 instead of cmor)
            params = tfa_params.copy()
            if tfa == "wpt":
                params["wavelet"] = "db4"

            matrix, freqs = time_freq_factory(signal_1d, fs, method=tfa, **params)
            print(f" [{matrix.shape[0]} bands]", end="", flush=True)
        except Exception as e:
            print(f" — FAILED: {e}")
            for bss in bss_methods:
                results[(tfa, bss)] = {"status": f"TFA_ERR: {e}"}
            continue

        # Run each BSS method
        for bss in bss_methods:
            try:
                S_est, A_est, W = bss_factory(matrix, method=bss, n_components=n_sources)
                indep = compute_independence_metric(S_est)
                ffds = compute_fault_detection_score(S_est, fs, fault_freqs)

                results[(tfa, bss)] = {
                    "S_est": S_est,
                    "independence": indep,
                    "ffds": ffds,
                    "status": "OK",
                }
            except Exception as e:
                results[(tfa, bss)] = {
                    "S_est": None,
                    "independence": float("nan"),
                    "ffds": float("nan"),
                    "status": f"ERROR: {str(e)[:40]}",
                }

        # Print summary for this TFA
        ffds_vals = [results[(tfa, b)]["ffds"] for b in bss_methods
                     if results[(tfa, b)].get("S_est") is not None]
        avg_ffds = np.mean(ffds_vals) if ffds_vals else 0
        print(f" — avg FFDS={avg_ffds:.1f}")

    return results


def plot_cross_heatmap(results, tfa_methods, bss_methods, sample_name, out_dir):
    """Plot TFA × BSS FFDS heatmap."""
    n_tfa = len(tfa_methods)
    n_bss = len(bss_methods)

    ffds_matrix = np.zeros((n_tfa, n_bss))
    for i, tfa in enumerate(tfa_methods):
        for j, bss in enumerate(bss_methods):
            val = results.get((tfa, bss), {}).get("ffds", 0)
            ffds_matrix[i, j] = val if not np.isnan(val) else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ffds_matrix, cmap="YlGn", aspect="auto")

    ax.set_xticks(range(n_bss))
    ax.set_xticklabels([m.upper() for m in bss_methods])
    ax.set_yticks(range(n_tfa))
    ax.set_yticklabels([m.upper() for m in tfa_methods])
    ax.set_xlabel("BSS Method")
    ax.set_ylabel("TFA Method")
    ax.set_title(f"FFDS Heatmap — {sample_name}")

    for i in range(n_tfa):
        for j in range(n_bss):
            ax.text(j, i, f"{ffds_matrix[i, j]:.1f}", ha="center", va="center", fontsize=10)

    # Highlight best
    best_idx = np.unravel_index(ffds_matrix.argmax(), ffds_matrix.shape)
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=3))

    plt.colorbar(im, ax=ax, label="FFDS")
    plt.tight_layout()

    path = os.path.join(out_dir, f"{sample_name}_ffds_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")

    return ffds_matrix


# ============================================================
# Visualization: Cross-Dataset Summary
# ============================================================

def plot_phase1_summary(all_phase1, out_dir):
    """Plot Phase 1 summary: computation time and entropy comparison."""
    datasets = sorted(all_phase1.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("TFA Method Characteristics (Phase 1)", fontsize=14, fontweight="bold")

    # Computation time
    ax = axes[0]
    methods = TFA_METHODS
    x = np.arange(len(datasets))
    width = 0.12
    for i, method in enumerate(methods):
        times = []
        for ds in datasets:
            ds_results = all_phase1[ds]
            vals = [r.get(method, {}).get("elapsed", 0) for r in ds_results.values()]
            times.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, times, width, label=method.upper())

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Time [s]")
    ax.set_title("Computation Time")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(datasets, rotation=15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Entropy (energy concentration)
    ax = axes[1]
    for i, method in enumerate(methods):
        entropies = []
        for ds in datasets:
            ds_results = all_phase1[ds]
            vals = [r.get(method, {}).get("entropy", 0) for r in ds_results.values()]
            entropies.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, entropies, width, label=method.upper())

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Entropy (lower = more concentrated)")
    ax.set_title("Energy Concentration")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(datasets, rotation=15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "phase1_characteristics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_phase2_summary(all_phase2, out_dir):
    """Plot Phase 2 summary: FFDS heatmap across datasets."""
    datasets = sorted(all_phase2.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle("TFA × BSS FFDS Heatmap (Phase 2)", fontsize=14, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        # Average across samples
        ffds_matrix = np.zeros((len(TFA_METHODS), len(BSS_METHODS)))
        count = 0
        for sample_name, results in all_phase2[ds].items():
            for i, tfa in enumerate(TFA_METHODS):
                for j, bss in enumerate(BSS_METHODS):
                    val = results.get((tfa, bss), {}).get("ffds", 0)
                    ffds_matrix[i, j] += val if not np.isnan(val) else 0
            count += 1
        if count > 0:
            ffds_matrix /= count

        im = ax.imshow(ffds_matrix, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(BSS_METHODS)))
        ax.set_xticklabels([m.upper() for m in BSS_METHODS])
        ax.set_yticks(range(len(TFA_METHODS)))
        ax.set_yticklabels([m.upper() for m in TFA_METHODS])
        ax.set_xlabel("BSS Method")
        ax.set_ylabel("TFA Method")
        ax.set_title(f"{ds.upper()}")

        for i in range(len(TFA_METHODS)):
            for j in range(len(BSS_METHODS)):
                ax.text(j, i, f"{ffds_matrix[i, j]:.1f}", ha="center", va="center", fontsize=9)

        best_idx = np.unravel_index(ffds_matrix.argmax(), ffds_matrix.shape)
        ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                    fill=False, edgecolor="gold", linewidth=3))
        plt.colorbar(im, ax=ax, label="FFDS")

    plt.tight_layout()
    path = os.path.join(out_dir, "phase2_cross_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def find_best_combinations(all_phase2):
    """Find best TFA+BSS combination for each dataset."""
    print("\n  Best TFA+BSS Combinations:")
    print(f"  {'Dataset':<12} {'Best TFA':<10} {'Best BSS':<10} {'FFDS':<10}")
    print(f"  {'-'*42}")

    for ds in sorted(all_phase2.keys()):
        best_ffds = 0
        best_combo = ("", "")
        for sample_name, results in all_phase2[ds].items():
            for (tfa, bss), metrics in results.items():
                ffds = metrics.get("ffds", 0)
                if not np.isnan(ffds) and ffds > best_ffds:
                    best_ffds = ffds
                    best_combo = (tfa, bss)
        print(f"  {ds:<12} {best_combo[0].upper():<10} {best_combo[1].upper():<10} {best_ffds:.1f}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("TFA Method Comparison (Two-Phase)")
    print("=" * 60)
    print(f"TFA Methods: {[m.upper() for m in TFA_METHODS]}")
    print(f"BSS Methods: {[m.upper() for m in BSS_METHODS]}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print()

    all_phase1 = {}  # ds -> {sample -> {tfa -> metrics}}
    all_phase2 = {}  # ds -> {sample -> {(tfa,bss) -> metrics}}
    report_rows = []

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
            print(f"  ERROR: No data loaded. Skipping.")
            continue

        all_phase1[ds_name] = {}
        all_phase2[ds_name] = {}

        for sample_name, signals, fs, fault_freqs in samples:
            print(f"\n  --- {sample_name} ---")

            # Preprocess
            signals_pre, fs_pre = preprocess_signals(signals, fs, ds_config["preprocess"])
            signal_1d = signals_pre[0]
            print(f"  Signal: {len(signal_1d)} samples @ {fs_pre} Hz")

            tfa_params = ds_config["tfa_params"]
            n_sources = ds_config["n_sources"]

            # Phase 1: TFA Characteristics
            print(f"\n  Phase 1: TFA Characteristics")
            phase1_results = {}
            for tfa in TFA_METHODS:
                res = evaluate_tfa_characteristics(signal_1d, fs_pre, tfa, tfa_params, fault_freqs)
                phase1_results[tfa] = res
                status = res["status"]
                elapsed = res["elapsed"]
                entropy = res["entropy"]
                print(f"    {tfa.upper():<10} time={elapsed:.2f}s entropy={entropy:.2f} [{status}]")

                report_rows.append({
                    "phase": "1",
                    "dataset": ds_name,
                    "sample": sample_name,
                    "tfa": tfa,
                    "bss": "-",
                    "elapsed": f"{elapsed:.3f}",
                    "entropy": f"{entropy:.3f}",
                    "ffds": "-",
                    "independence": "-",
                    "status": status,
                })

            all_phase1[ds_name][sample_name] = phase1_results

            # Plot spectrograms
            plot_tfa_spectrograms(phase1_results, fs_pre, sample_name, ds_out)

            # Phase 2: TFA × BSS Cross-Comparison
            print(f"\n  Phase 2: TFA × BSS Cross-Comparison")
            phase2_results = run_cross_comparison(
                signal_1d, fs_pre, TFA_METHODS, BSS_METHODS,
                tfa_params, n_sources, fault_freqs, ds_out, sample_name,
            )
            all_phase2[ds_name][sample_name] = phase2_results

            # Save results
            for (tfa, bss), metrics in phase2_results.items():
                report_rows.append({
                    "phase": "2",
                    "dataset": ds_name,
                    "sample": sample_name,
                    "tfa": tfa,
                    "bss": bss,
                    "elapsed": "-",
                    "entropy": "-",
                    "ffds": f"{metrics.get('ffds', 0):.2f}" if metrics.get("S_est") is not None else "-",
                    "independence": f"{metrics.get('independence', 0):.4f}" if metrics.get("S_est") is not None else "-",
                    "status": metrics.get("status", "UNKNOWN"),
                })

            # Plot heatmap
            plot_cross_heatmap(phase2_results, TFA_METHODS, BSS_METHODS, sample_name, ds_out)

    # ============================================================
    # Cross-Dataset Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("Generating Cross-Dataset Summary")
    print(f"{'='*60}")

    if all_phase1:
        plot_phase1_summary(all_phase1, OUTPUT_DIR)

    if all_phase2:
        plot_phase2_summary(all_phase2, OUTPUT_DIR)
        find_best_combinations(all_phase2)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    keys = ["phase", "dataset", "sample", "tfa", "bss", "elapsed", "entropy", "ffds", "independence", "status"]
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
