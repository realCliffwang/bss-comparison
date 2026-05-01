"""
CWRU Bearing Fault Detection: CWT + BSS End-to-End Experiment
=============================================================

PIPELINE:
  load_cwru -> preprocess -> CWT -> build_observation_matrix -> run_bss -> evaluate

SETUP (TODO — you need to do these before running):
  1. Download CWRU .mat files from https://zenodo.org/records/10987113
  2. Extract all .mat files into:  data/cwru/
  3. Verify filenames match the mapping in src/data_loader.py (CWRU_FAULT_MAP).
     Run list_cwru_files() to check what's available.
  4. Adjust CONFIG below as needed.

FOR YOUR STAMPING DIE DATA:
  - Replace load_cwru() call with your own data loader
  - Adjust fs, freq_range, n_bands, feature_freqs in CONFIG
  - The rest of the pipeline (preprocess -> CWT -> BSS -> evaluate) stays the same
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" — change if needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_cwru, list_cwru_files
from preprocessing import preprocess_signals
from cwt_module import cwt_transform, build_observation_matrix, plot_cwt_spectrogram
from bss_module import run_bss
from evaluation import (
    evaluate_bss,
    plot_waveform_comparison,
    plot_spectrum_comparison,
    plot_envelope_spectrum,
    plot_correlation_matrix,
)

# ============================================================
# CONFIGURATION — Adjust these values for your experiment
# ============================================================

CONFIG = {
    # Data selection
    "data_dir": "data/cwru",
    "fault_type": "inner_race_007",  # inner_race_007, ball_007, outer_race_6_007, normal
    "load": 0,                        # 0, 1, 2, 3
    "channels": ["DE"],               # ["DE"] for single-channel, ["DE","FE"] for dual

    # Preprocessing
    "preprocess": {
        "detrend": True,
        "bandpass": (100, 5000),      # Bandpass filter range (Hz). None = no filter.
        "normalize": "zscore",        # "zscore", "minmax", or None
        "resample_fs": None,          # None = no resampling
    },

    # CWT + Observation Matrix
    "cwt": {
        "wavelet": "cmor1.5-1.0",
        "n_bands": 20,                # Number of frequency bands
        "freq_range": (100, 5000),    # Frequency range for CWT (Hz)
        "mode": "single_channel_expansion",  # or "multi_channel"
    },

    # BSS
    "bss": {
        "method": "SOBI",             # "SOBI" or "FastICA"
        "n_sources": 5,
        "n_lags": 50,                 # SOBI only
    },

    # Evaluation — bearing characteristic frequencies
    # TODO: Calculate these from your bearing geometry and shaft speed!
    # Example values for 6205-2RS bearing at ~1797 RPM:
    #   BPFO ≈ 3.585 × shaft_freq  (outer race)
    #   BPFI ≈ 5.415 × shaft_freq  (inner race)
    #   BSF  ≈ 2.358 × shaft_freq  (ball spin)
    #   FTF  ≈ 0.398 × shaft_freq  (cage)
    "feature_freqs": {
        "BPFO": 107.3,    # TODO: recalculate for your data
        "BPFI": 162.2,
        "BSF":  70.6,
    },

    # Plotting
    "output_dir": "outputs/cwru",     # Where to save figures
}


def main():
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("=" * 60)
    print("CWRU CWT + BSS Experiment")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading CWRU data...")
    signals, fs, rpm = load_cwru(
        data_dir=CONFIG["data_dir"],
        fault_type=CONFIG["fault_type"],
        load=CONFIG["load"],
        channels=CONFIG["channels"],
    )
    print(f"  Loaded {signals.shape[0]} channel(s), {signals.shape[1]} samples @ {fs} Hz")
    print(f"  Duration: {signals.shape[1]/fs:.2f} s, RPM: {rpm}")

    # Use a subset for faster processing (first 2 seconds)
    n_use = min(signals.shape[1], int(2.0 * fs))
    signals = signals[:, :n_use]
    print(f"  Using first {n_use} samples ({n_use/fs:.2f} s) for fast demo")

    # Step 2: Preprocess
    print("\n[2/6] Preprocessing...")
    signals_pre, fs_pre = preprocess_signals(signals, fs, CONFIG["preprocess"])
    print(f"  Preprocessed shape: {signals_pre.shape}, fs: {fs_pre} Hz")

    # Step 3: CWT + Build observation matrix
    print("\n[3/6] Computing CWT and building observation matrix...")
    X_for_bss, obs_labels = build_observation_matrix(
        signals_pre, fs_pre, CONFIG["cwt"]
    )
    print(f"  Observation matrix: {X_for_bss.shape[0]} obs × {X_for_bss.shape[1]} samples")
    print(f"  First few labels: {obs_labels[:5]}")

    # Visualize CWT of first channel
    coef, freqs, scales = cwt_transform(signals_pre[0], fs_pre,
                                        wavelet=CONFIG["cwt"]["wavelet"],
                                        n_bands=CONFIG["cwt"]["n_bands"],
                                        freq_range=CONFIG["cwt"]["freq_range"])
    fig, _ = plot_cwt_spectrogram(coef, freqs, signals_pre.shape[1], fs_pre,
                                   title=f"CWRU {CONFIG['fault_type']} — CWT Spectrogram (DE)")
    fig.savefig(os.path.join(CONFIG["output_dir"], "cwt_spectrogram.png"), dpi=150)
    plt.close(fig)

    # Step 4: BSS
    print(f"\n[4/6] Running BSS ({CONFIG['bss']['method']})...")
    S_est, W = run_bss(X_for_bss, **CONFIG["bss"])
    print(f"  Estimated sources: {S_est.shape[0]} × {S_est.shape[1]}")

    # Step 5: Evaluation
    print("\n[5/6] Evaluating...")
    evaluate_bss(S_est, W, X_for_bss, fs_pre, CONFIG)

    # Waveform comparison (first few sources vs first few observations)
    fig, _ = plot_waveform_comparison(
        X_for_bss, S_est, fs_pre, max_duration=0.5,
        title_prefix=f"CWRU {CONFIG['fault_type']} — "
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "waveform_comparison.png"), dpi=150)
    plt.close(fig)

    # Spectrum comparison
    fig, _ = plot_spectrum_comparison(
        X_for_bss, S_est, fs_pre,
        title_prefix=f"CWRU {CONFIG['fault_type']} — "
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "spectrum_comparison.png"), dpi=150)
    plt.close(fig)

    # Envelope spectrum with fault frequency markers
    labels = [f"Src {i}" for i in range(S_est.shape[0])]
    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre,
        fault_freqs=CONFIG["feature_freqs"],
        labels=labels,
        title_prefix=f"CWRU {CONFIG['fault_type']} — "
    )
    fig.savefig(os.path.join(CONFIG["output_dir"], "envelope_spectrum.png"), dpi=150)
    plt.close(fig)

    # Source correlation matrix
    fig, _ = plot_correlation_matrix(S_est, title="CWRU Source Correlation")
    fig.savefig(os.path.join(CONFIG["output_dir"], "correlation_matrix.png"), dpi=150)
    plt.close(fig)

    print("\n[6/6] Done! Results saved to:", CONFIG["output_dir"])
    print("=" * 60)


if __name__ == "__main__":
    main()
