"""
Quick Start Example: CWRU Bearing Fault Diagnosis

This example demonstrates the basic BSS-Test workflow:
1. Load data
2. Preprocess signals
3. Build observation matrix (CWT)
4. Run BSS (SOBI)
5. Evaluate and visualize results

Usage:
    python examples/quick_start.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test import preprocess_signals, ExperimentConfig
from bss_test.io.cwru import load_cwru
from bss_test.tfa import build_observation_matrix
from bss_test.tfa.cwt import cwt_transform
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    evaluate_bss,
    plot_waveform_comparison,
    plot_spectrum_comparison,
    plot_envelope_spectrum,
    plot_correlation_matrix,
)
from bss_test.utils.logger import setup_logging, get_logger
from bss_test.utils.synthetic import generate_synthetic_mixture

setup_logging(level="info")
logger = get_logger(__name__)


def main():
    config = ExperimentConfig.from_yaml("configs/cwru.yaml")

    output_dir = "outputs/examples/quick_start"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BSS-Test Quick Start Example")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n[1/5] Loading CWRU data...")
    try:
        signals, fs, rpm = load_cwru(
            data_dir=config.data_dir,
            fault_type=config.fault_type,
            load=config.load,
            channels=config.channels,
        )
        logger.info(f"  Loaded {signals.shape[0]} channel(s), {signals.shape[1]} samples @ {fs} Hz")
    except FileNotFoundError:
        logger.warning("  CWRU data not found. Using synthetic data instead.")
        signals_true, signals, _ = generate_synthetic_mixture(
            n_sources=3, n_obs=5, n_samples=20000, fs=1000
        )
        fs = 1000

    n_use = min(signals.shape[1], int(2.0 * fs))
    signals = signals[:, :n_use]
    logger.info(f"  Using first {n_use} samples ({n_use/fs:.2f} s)")

    # Step 2: Preprocess
    logger.info("\n[2/5] Preprocessing...")
    preprocess_config = {
        "detrend": config.preprocess.detrend,
        "bandpass": config.preprocess.bandpass,
        "normalize": config.preprocess.normalize,
    }
    signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)
    logger.info(f"  Preprocessed shape: {signals_pre.shape}, fs: {fs_pre} Hz")

    # Step 3: CWT + Observation Matrix
    logger.info("\n[3/5] Computing CWT and building observation matrix...")
    tfa_config = {
        "mode": config.tfa.mode,
        "tfa_method": config.tfa.tfa_method,
        "n_bands": config.tfa.n_bands,
        "freq_range": config.tfa.freq_range,
        "wavelet": config.tfa.wavelet,
    }
    X_for_bss, obs_labels = build_observation_matrix(signals_pre, fs_pre, tfa_config)
    logger.info(f"  Observation matrix: {X_for_bss.shape[0]} obs x {X_for_bss.shape[1]} samples")

    # Step 4: BSS
    logger.info(f"\n[4/5] Running BSS ({config.bss.method})...")
    S_est, A_est, W = bss_factory(
        X_for_bss,
        method=config.bss.method,
        n_components=config.bss.n_sources,
    )
    logger.info(f"  Estimated sources: {S_est.shape[0]} x {S_est.shape[1]}")

    # Step 5: Evaluation
    logger.info("\n[5/5] Evaluating...")
    evaluate_bss(S_est, W, X_for_bss, fs_pre, config)

    fig, _ = plot_waveform_comparison(X_for_bss, S_est, fs_pre, max_duration=0.5,
                                       title_prefix="Quick Start — ")
    fig.savefig(os.path.join(output_dir, "waveform_comparison.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_envelope_spectrum(S_est, fs_pre, fault_freqs=config.feature_freqs,
                                     title_prefix="Quick Start — ")
    fig.savefig(os.path.join(output_dir, "envelope_spectrum.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_correlation_matrix(S_est, title="Source Correlation")
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)

    logger.info(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
