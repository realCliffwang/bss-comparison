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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.data_loader import load_cwru
from src.preprocessing import preprocess_signals
from src.cwt_module import cwt_transform, build_observation_matrix, plot_cwt_spectrogram
from src.bss_module import run_bss
from src.evaluation import (
    evaluate_bss,
    plot_waveform_comparison,
    plot_spectrum_comparison,
    plot_envelope_spectrum,
    plot_correlation_matrix,
)
from src.config import get_config
from src.logger import setup_logging, get_logger

# Setup logging
setup_logging(level="info")
logger = get_logger(__name__)


def main():
    """Run quick start example."""

    # Get default configuration
    config = get_config("cwru")

    # Create output directory
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
        from src.utils import generate_synthetic_mixture
        signals_true, signals, _ = generate_synthetic_mixture(
            n_sources=3, n_obs=5, n_samples=20000, fs=1000
        )
        fs = 1000

    # Use subset for demo
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
    cwt_config = {
        "mode": config.cwt.mode,
        "tfa_method": config.cwt.tfa_method,
        "n_bands": config.cwt.n_bands,
        "freq_range": config.cwt.freq_range,
        "wavelet": config.cwt.wavelet,
    }
    X_for_bss, obs_labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
    logger.info(f"  Observation matrix: {X_for_bss.shape[0]} obs × {X_for_bss.shape[1]} samples")

    # Plot CWT spectrogram
    coef, freqs, scales = cwt_transform(
        signals_pre[0], fs_pre,
        wavelet=config.cwt.wavelet,
        n_bands=config.cwt.n_bands,
        freq_range=config.cwt.freq_range,
    )
    fig, _ = plot_cwt_spectrogram(coef, freqs, signals_pre.shape[1], fs_pre,
                                   title="CWT Spectrogram (Channel 0)")
    fig.savefig(os.path.join(output_dir, "cwt_spectrogram.png"), dpi=150)
    plt.close(fig)

    # Step 4: BSS
    logger.info(f"\n[4/5] Running BSS ({config.bss.method})...")
    S_est, W = run_bss(
        X_for_bss,
        method=config.bss.method,
        n_sources=config.bss.n_sources,
        n_lags=config.bss.n_lags,
    )
    logger.info(f"  Estimated sources: {S_est.shape[0]} × {S_est.shape[1]}")

    # Step 5: Evaluation
    logger.info("\n[5/5] Evaluating...")
    evaluate_bss(S_est, W, X_for_bss, fs_pre, config)

    # Generate plots
    fig, _ = plot_waveform_comparison(
        X_for_bss, S_est, fs_pre, max_duration=0.5,
        title_prefix="Quick Start — "
    )
    fig.savefig(os.path.join(output_dir, "waveform_comparison.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_spectrum_comparison(
        X_for_bss, S_est, fs_pre,
        title_prefix="Quick Start — "
    )
    fig.savefig(os.path.join(output_dir, "spectrum_comparison.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre,
        fault_freqs=config.feature_freqs,
        title_prefix="Quick Start — "
    )
    fig.savefig(os.path.join(output_dir, "envelope_spectrum.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_correlation_matrix(S_est, title="Source Correlation")
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)

    logger.info(f"\nDone! Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
