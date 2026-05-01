"""
PHM 2010 Milling Tool Wear: CWT + BSS — All 6 Cutters
======================================================
c1, c4, c6: training data (with wear labels) → full analysis + wear evolution
c2, c3, c5: test data (no wear) → CWT + BSS separation only

Outputs saved to outputs/phm2010/c{id}/ per cutter.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_phm_cut, load_phm_wear
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

CONFIG = {
    "data_dir": "data/phm2010_milling",
    "sensor_types": ["vib_x", "vib_y", "vib_z"],
    "downsample": 1,
    "preprocess": {
        "detrend": True,
        "bandpass": (100, 20000),
        "normalize": "zscore",
        "resample_fs": None,
    },
    "cwt": {
        "wavelet": "cmor1.5-1.0",
        "n_bands": 20,
        "freq_range": (100, 20000),
        "bands_per_ch": 7,
        "mode": "multi_channel",
    },
    "bss": {
        "method": "SOBI",
        "n_sources": 6,
        "n_lags": 50,
    },
    "base_output": "outputs/phm2010",
}


def process_cutter(tool_id, is_training):
    out_dir = os.path.join(CONFIG["base_output"], f"c{tool_id}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CUTTER c{tool_id}" + (" (TRAINING + wear)" if is_training else " (TEST)"))
    print(f"{'='*60}")

    # --- Load single representative cut ---
    rep_cut = 150  # middle of run
    print(f"[1/4] Loading cut {rep_cut}...")
    signals, fs, cn = load_phm_cut(
        tool_id=tool_id, cut_no=rep_cut,
        data_dir=CONFIG["data_dir"],
        sensor_types=CONFIG["sensor_types"],
        downsample=CONFIG["downsample"],
    )
    n_use = min(signals.shape[1], int(1.0 * fs))
    signals = signals[:, :n_use]
    print(f"  {signals.shape[0]} ch x {n_use} samples @ {fs} Hz ({n_use/fs:.1f}s)")

    # --- Preprocess ---
    print("[2/4] Preprocessing...")
    signals_pre, fs_pre = preprocess_signals(signals, fs, CONFIG["preprocess"])

    # --- CWT + BSS ---
    print("[3/4] CWT + BSS (SOBI)...")
    X_for_bss, obs_labels = build_observation_matrix(signals_pre, fs_pre, CONFIG["cwt"])
    S_est, W = run_bss(X_for_bss, **CONFIG["bss"])
    print(f"  Obs: {X_for_bss.shape[0]} x {X_for_bss.shape[1]}, Src: {S_est.shape[0]} x {S_est.shape[1]}")

    # --- Save plots ---
    print("[4/4] Generating plots...")

    # CWT spectrogram
    coef, freqs, scales = cwt_transform(
        signals_pre[0], fs_pre, wavelet=CONFIG["cwt"]["wavelet"],
        n_bands=CONFIG["cwt"]["n_bands"], freq_range=CONFIG["cwt"]["freq_range"],
    )
    fig, _ = plot_cwt_spectrogram(coef, freqs, signals_pre.shape[1], fs_pre,
                                   title=f"PHM c{tool_id} Cut {rep_cut} — CWT")
    fig.savefig(os.path.join(out_dir, "cwt_spectrogram.png"), dpi=150)
    plt.close(fig)

    # Waveform comparison
    fig, _ = plot_waveform_comparison(
        X_for_bss, S_est, fs_pre, max_duration=0.3,
        title_prefix=f"c{tool_id} Cut {rep_cut} — ")
    fig.savefig(os.path.join(out_dir, "waveform_comparison.png"), dpi=150)
    plt.close(fig)

    # Spectrum comparison
    fig, _ = plot_spectrum_comparison(
        X_for_bss, S_est, fs_pre,
        title_prefix=f"c{tool_id} Cut {rep_cut} — ")
    fig.savefig(os.path.join(out_dir, "spectrum_comparison.png"), dpi=150)
    plt.close(fig)

    # Envelope spectrum
    labels = [f"Src {i}" for i in range(S_est.shape[0])]
    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre, labels=labels,
        title_prefix=f"c{tool_id} Cut {rep_cut} — ")
    fig.savefig(os.path.join(out_dir, "envelope_spectrum.png"), dpi=150)
    plt.close(fig)

    # Correlation matrix
    fig, _ = plot_correlation_matrix(S_est, title=f"c{tool_id} — Source Correlation")
    fig.savefig(os.path.join(out_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)

    # --- Wear evolution (training only) ---
    if is_training:
        print(f"  Wear evolution...")
        wear = load_phm_wear(tool_id, CONFIG["data_dir"])
        wear = wear[~np.isnan(wear)]
        print(f"    {len(wear)} labeled runs")

        # Sample cuts evenly distributed across tool life
        n_wear_pts = 15
        sample_cuts = np.linspace(1, len(wear), n_wear_pts, dtype=int)
        S_list = []
        valid_wear = []

        for cut_no in sample_cuts:
            try:
                sig_cut, fs_cut, _ = load_phm_cut(
                    tool_id=tool_id, cut_no=cut_no,
                    data_dir=CONFIG["data_dir"],
                    sensor_types=CONFIG["sensor_types"],
                    downsample=CONFIG["downsample"],
                )
                n_u = min(sig_cut.shape[1], int(0.5 * fs_cut))
                sig_cut = sig_cut[:, :n_u]
                sig_p, fs_p = preprocess_signals(sig_cut, fs_cut, CONFIG["preprocess"])
                X_obs, _ = build_observation_matrix(sig_p, fs_p, CONFIG["cwt"])
                S_run, _ = run_bss(X_obs, **CONFIG["bss"])
                S_list.append(S_run)
                valid_wear.append(wear[cut_no - 1])
                print(f"    cut {cut_no:3d} VB={wear[cut_no-1]:.1f}")
            except Exception as e:
                print(f"    cut {cut_no:3d} SKIPPED: {e}")

        if len(S_list) > 2:
            fig, _ = plot_wear_evolution(
                S_list, np.array(valid_wear), tool_id=tool_id,
                title_prefix="PHM 2010 ")
            fig.savefig(os.path.join(out_dir, "wear_evolution.png"), dpi=150)
            plt.close(fig)

    n_files = len(os.listdir(out_dir))
    print(f"DONE c{tool_id} → {n_files} plots in {out_dir}")


def main():
    print("=" * 60)
    print("PHM 2010 — ALL 6 CUTTERS")
    print("=" * 60)

    for tool_id in range(1, 7):
        is_training = tool_id in [1, 4, 6]
        process_cutter(tool_id, is_training)

    print(f"\n{'='*60}")
    print(f"ALL DONE — results in {CONFIG['base_output']}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
