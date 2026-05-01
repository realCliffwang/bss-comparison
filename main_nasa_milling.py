"""
NASA / UC Berkeley Milling Tool Wear: CWT + BSS End-to-End Experiment
=====================================================================

This script is a drop-in replacement for main_phm_milling.py, using the
NASA/UC Berkeley milling dataset (mill.mat) which has been successfully
downloaded and verified.

PIPELINE:
  load_nasa_milling -> preprocess -> CWT -> build_obs_matrix -> run_bss -> evaluate

Original PHM 2010 CDN is dead — this NASA dataset provides equivalent
multi-channel milling vibration + tool wear data.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_nasa_milling, load_nasa_milling_single
from preprocessing import preprocess_signals
from cwt_module import cwt_transform, build_observation_matrix, plot_cwt_spectrogram
from bss_module import run_bss
from evaluation import (
    evaluate_bss,
    plot_waveform_comparison,
    plot_spectrum_comparison,
    plot_envelope_spectrum,
    plot_correlation_matrix,
    plot_wear_evolution,
)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "data_dir": "data/phm2010_milling",

    # Sensor channels (subset of: force_ac, force_dc, vib_table, vib_spindle, AE_table, AE_spindle)
    "sensor_types": ["vib_table", "vib_spindle", "force_ac"],

    # Preprocessing
    "preprocess": {
        "detrend": True,
        "bandpass": (10, 100),         # 10-100 Hz covers spindle freq and harmonics
        "normalize": "zscore",
        "resample_fs": None,
    },

    # CWT + Observation Matrix
    "cwt": {
        "wavelet": "cmor1.5-1.0",
        "n_bands": 20,
        "freq_range": (10, 120),
        "bands_per_ch": 7,             # 3 ch × 7 ≈ 21 obs
        "mode": "multi_channel",
    },

    # BSS
    "bss": {
        "method": "SOBI",
        "n_sources": 6,
        "n_lags": 50,
    },

    # Output
    "output_dir": "outputs/nasa_milling",
}


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("=" * 60)
    print("NASA Milling CWT + BSS Experiment")
    print("=" * 60)

    # ---- Single Run Demo ----
    print("\n[1/5] Loading single run (case 1, run 12, VB~0.40)...")
    # Find a run with significant wear
    all_signals_temp, all_meta_temp, fs = load_nasa_milling(
        CONFIG["data_dir"],
        sensor_types=CONFIG["sensor_types"],
        case_filter=[1],
        include_wear_only=True,
    )
    # Pick a run with moderate wear
    demo_idx = min(10, len(all_signals_temp) - 1)
    signals, meta = all_signals_temp[demo_idx], all_meta_temp[demo_idx]
    print(f"  Case {meta['case']}, Run {meta['run']}, VB={meta['VB']:.3f} mm")
    print(f"  Signals: {signals.shape[0]} ch × {signals.shape[1]} samples @ {fs} Hz")

    print("\n[2/5] Preprocessing...")
    signals_pre, fs_pre = preprocess_signals(signals, fs, CONFIG["preprocess"])
    print(f"  Preprocessed: {signals_pre.shape}, fs={fs_pre} Hz")

    print("\n[3/5] CWT + observation matrix...")
    X_for_bss, obs_labels = build_observation_matrix(
        signals_pre, fs_pre, CONFIG["cwt"]
    )
    print(f"  X_for_bss: {X_for_bss.shape[0]} obs × {X_for_bss.shape[1]} samples")

    # CWT spectrogram of first channel
    coef, freqs, scales = cwt_transform(
        signals_pre[0], fs_pre,
        wavelet=CONFIG["cwt"]["wavelet"],
        n_bands=CONFIG["cwt"]["n_bands"],
        freq_range=CONFIG["cwt"]["freq_range"],
    )
    fig, _ = plot_cwt_spectrogram(coef, freqs, signals_pre.shape[1], fs_pre,
                                   title="NASA Milling — CWT Spectrogram (Vib Table)")
    fig.savefig(os.path.join(CONFIG["output_dir"], "cwt_spectrogram.png"), dpi=150)
    plt.close(fig)

    print(f"\n[4/5] Running BSS ({CONFIG['bss']['method']})...")
    S_est, W = run_bss(X_for_bss, **CONFIG["bss"])
    print(f"  Sources: {S_est.shape[0]} × {S_est.shape[1]}")

    # ---- Evaluation Plots ----
    print("\n[5/5] Generating evaluation plots...")

    fig, _ = plot_waveform_comparison(
        X_for_bss, S_est, fs_pre, max_duration=5.0,
        title_prefix=f"NASA Mill Case {meta['case']} Run {meta['run']} — "
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "waveform_comparison.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_spectrum_comparison(
        X_for_bss, S_est, fs_pre,
        title_prefix=f"NASA Mill Case {meta['case']} Run {meta['run']} — "
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "spectrum_comparison.png"), dpi=150)
    plt.close(fig)

    # Envelope spectrum of separated sources
    labels = [f"Src {i}" for i in range(S_est.shape[0])]
    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre, labels=labels,
        title_prefix=f"NASA Mill Case {meta['case']} Run {meta['run']} — "
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "envelope_spectrum.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_correlation_matrix(
        S_est, title=f"NASA Milling — Source Correlation (Case {meta['case']})"
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "correlation_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- Multi-Run Wear Evolution (Case 1) ----
    print("\n--- Wear evolution analysis (Case 1, all runs) ---")
    all_signals, all_meta, fs = load_nasa_milling(
        CONFIG["data_dir"],
        sensor_types=CONFIG["sensor_types"],
        case_filter=[1],
        include_wear_only=True,
    )
    wear_values = np.array([m["VB"] for m in all_meta])
    print(f"  {len(all_signals)} runs, VB range: [{wear_values.min():.3f}, {wear_values.max():.3f}]")

    S_list = []
    valid_wear = []
    for i, (sig_run, meta_run) in enumerate(zip(all_signals, all_meta)):
        if np.isnan(meta_run["VB"]):
            continue
        try:
            sig_p, fs_p = preprocess_signals(sig_run, fs, CONFIG["preprocess"])
            X_obs, _ = build_observation_matrix(sig_p, fs_p, CONFIG["cwt"])
            S_run, _ = run_bss(X_obs, **CONFIG["bss"])
            S_list.append(S_run)
            valid_wear.append(meta_run["VB"])
        except Exception as e:
            print(f"  Skipped run {meta_run['run']}: {e}")

    if len(S_list) > 2:
        fig, _ = plot_wear_evolution(
            S_list, np.array(valid_wear), tool_id=1,
            title_prefix="NASA Milling "
        )
        fig.savefig(os.path.join(CONFIG["output_dir"], "wear_evolution.png"), dpi=150)
        plt.close(fig)

    print(f"\nDone! Results saved to: {CONFIG['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
