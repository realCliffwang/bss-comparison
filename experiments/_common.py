"""
Shared utilities for experiment scripts.

Eliminates code duplication across experiments/comparison/ and experiments/single/
by providing common functions for data loading, preprocessing, training, evaluation,
plotting, and CSV output.
"""

import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test import preprocess_signals, ExperimentConfig
from bss_test.feature_extractor import extract_time_domain_features, extract_freq_domain_features
from bss_test.evaluation import (
    plot_classifier_comparison,
    plot_confusion_matrix_grid,
    setup_academic_style,
)
from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Config helpers
# ============================================================

def build_preprocess_config(config):
    """Build preprocess dict from ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    dict
    """
    return {
        "detrend": config.preprocess.detrend,
        "bandpass": config.preprocess.bandpass,
        "normalize": config.preprocess.normalize,
    }


# ============================================================
# Data loading
# ============================================================

def load_cwru_signals(config, fault_type="inner_race_007", max_duration=2.0):
    """Load and preprocess a single CWRU signal.

    Returns
    -------
    signals_pre : ndarray (n_channels, n_samples)
    fs : float
    """
    from bss_test.io.cwru import load_cwru

    signals, fs, rpm = load_cwru(
        data_dir=config.data_dir,
        fault_type=fault_type,
        load=0,
        channels=["DE"],
    )
    n_use = min(signals.shape[1], int(max_duration * fs))
    signals = signals[:, :n_use]
    signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))
    return signals_pre, fs_pre


def load_nasa_signals(config, run_index=0, max_duration=5.0):
    """Load and preprocess a single NASA milling signal.

    Returns
    -------
    signals_pre : ndarray (n_channels, n_samples)
    fs : float
    meta : dict
    """
    from bss_test.io.nasa import load_nasa_milling_single

    signals, meta, fs = load_nasa_milling_single(
        run_index=run_index,
        data_dir=config.data_dir,
        sensor_types=["vib_table", "vib_spindle", "force_ac"],
    )
    n_use = min(signals.shape[1], int(max_duration * fs))
    signals = signals[:, :n_use]
    signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))
    return signals_pre, fs_pre, meta


def load_phm_signals(config, tool_id, cut_no=150, max_duration=1.0):
    """Load and preprocess a single PHM 2010 cut.

    Returns
    -------
    signals_pre : ndarray (n_channels, n_samples)
    fs : float
    """
    from bss_test.io.phm import load_phm_cut

    signals, fs, _ = load_phm_cut(
        tool_id=tool_id,
        cut_no=cut_no,
        data_dir=config.data_dir,
        sensor_types=["vib_x", "vib_y", "vib_z"],
    )
    n_use = min(signals.shape[1], int(max_duration * fs))
    signals = signals[:, :n_use]
    signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))
    return signals_pre, fs_pre


def load_cwru_features(config, fault_types=None, seg_sec=0.25, overlap=0.5):
    """Load CWRU multi-class data, segment, and extract features.

    Returns
    -------
    X : ndarray (n_segments, n_features)
    y : ndarray (n_segments,)
    """
    from bss_test.io.cwru import load_cwru

    if fault_types is None:
        fault_types = ["normal", "inner_race_007", "ball_007", "outer_race_6_007"]

    all_features = []
    all_labels = []

    for fault_type in fault_types:
        try:
            signals, fs, rpm = load_cwru(
                data_dir=config.data_dir,
                fault_type=fault_type,
                load=0,
                channels=["DE"],
            )
            n_use = min(signals.shape[1], int(2.0 * fs))
            signals = signals[:, :n_use]
            signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))

            n_segments = _extract_features_from_segments(
                signals_pre, fs_pre, fault_type, seg_sec, overlap,
                all_features, all_labels,
            )
            logger.info(f"  {fault_type}: {n_segments} segments")
        except Exception as e:
            logger.warning(f"  {fault_type} failed: {e}")

    return np.array(all_features), np.array(all_labels)


def load_phm_features(config, tool_labels=None, seg_sec=0.2, overlap=0.5):
    """Load PHM 2010 multi-tool data, segment, and extract features.

    Returns
    -------
    X : ndarray (n_segments, n_features)
    y : ndarray (n_segments,)
    """
    from bss_test.io.phm import load_phm_cut

    if tool_labels is None:
        tool_labels = {1: "low_wear", 4: "medium_wear", 6: "high_wear"}

    all_features = []
    all_labels = []

    for tool_id, label in tool_labels.items():
        try:
            signals, fs, cut_no = load_phm_cut(
                tool_id=tool_id,
                cut_no=150,
                data_dir=config.data_dir,
                sensor_types=["vib_x", "vib_y", "vib_z"],
            )
            n_use = min(signals.shape[1], int(1.0 * fs))
            signals = signals[:, :n_use]
            signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))

            n_segments = _extract_features_from_segments(
                signals_pre, fs_pre, label, seg_sec, overlap,
                all_features, all_labels,
            )
            logger.info(f"  c{tool_id} ({label}): {n_segments} segments")
        except Exception as e:
            logger.warning(f"  c{tool_id} failed: {e}")

    return np.array(all_features), np.array(all_labels)


def load_cwru_raw_segments(config, fault_types=None, segment_length=1024, overlap=0.5):
    """Load CWRU data and segment for WDCNN input.

    Returns
    -------
    X : ndarray (n_segments, segment_length)
    y : ndarray (n_segments,)
    """
    from bss_test.io.cwru import load_cwru
    from bss_test.wdcnn import segment_signals

    if fault_types is None:
        fault_types = ["normal", "inner_race_007", "ball_007", "outer_race_6_007"]

    all_segments = []
    all_labels = []

    for fault_type in fault_types:
        try:
            signals, fs, rpm = load_cwru(
                data_dir=config.data_dir,
                fault_type=fault_type,
                load=0,
                channels=["DE"],
            )
            n_use = min(signals.shape[1], int(2.0 * fs))
            signals = signals[:, :n_use]
            signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))

            X_seg, y_seg = segment_signals(
                signals_pre, np.array([fault_type]),
                segment_length=segment_length, overlap=overlap,
            )
            all_segments.append(X_seg)
            all_labels.append(y_seg)
            logger.info(f"  {fault_type}: {len(X_seg)} segments")
        except Exception as e:
            logger.warning(f"  {fault_type} failed: {e}")

    if not all_segments:
        return np.array([]), np.array([])

    return np.concatenate(all_segments), np.concatenate(all_labels)


def load_phm_raw_segments(config, tool_labels=None, segment_length=1024, overlap=0.5):
    """Load PHM 2010 data and segment for WDCNN input.

    Returns
    -------
    X : ndarray (n_segments, segment_length)
    y : ndarray (n_segments,)
    """
    from bss_test.io.phm import load_phm_cut
    from bss_test.wdcnn import segment_signals

    if tool_labels is None:
        tool_labels = {1: "low_wear", 4: "medium_wear", 6: "high_wear"}

    all_segments = []
    all_labels = []

    for tool_id, label in tool_labels.items():
        try:
            signals, fs, cut_no = load_phm_cut(
                tool_id=tool_id,
                cut_no=150,
                data_dir=config.data_dir,
                sensor_types=["vib_x", "vib_y", "vib_z"],
            )
            n_use = min(signals.shape[1], int(1.0 * fs))
            signals = signals[:, :n_use]
            signals_pre, fs_pre = preprocess_signals(signals, fs, build_preprocess_config(config))

            X_seg, y_seg = segment_signals(
                signals_pre, np.array([label]),
                segment_length=segment_length, overlap=overlap,
            )
            all_segments.append(X_seg)
            all_labels.append(y_seg)
            logger.info(f"  c{tool_id} ({label}): {len(X_seg)} segments")
        except Exception as e:
            logger.warning(f"  c{tool_id} failed: {e}")

    if not all_segments:
        return np.array([]), np.array([])

    return np.concatenate(all_segments), np.concatenate(all_labels)


# ============================================================
# Training & evaluation
# ============================================================

def run_classifier_comparison(train_fn, evaluate_fn, X_train, y_train,
                               X_test, y_test, methods):
    """Run timed training + evaluation for multiple classifiers.

    Parameters
    ----------
    train_fn : callable
        train_fn(X_train, y_train, method=str) -> model
    evaluate_fn : callable
        evaluate_fn(model, X_test, y_test) -> dict with accuracy, f1_macro, etc.
    X_train, y_train, X_test, y_test : array-like
    methods : list of str

    Returns
    -------
    list of dict
        Each dict: method, accuracy, f1_macro, confusion_matrix, label_names, time
    """
    results = []
    for method in methods:
        logger.info(f"\n    --- {method.upper()} ---")
        try:
            start_time = time.time()
            model = train_fn(X_train, y_train, method=method)
            elapsed = time.time() - start_time

            metrics = evaluate_fn(model, X_test, y_test)

            results.append({
                "method": method,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "confusion_matrix": metrics["confusion_matrix"],
                "label_names": metrics["label_names"],
                "time": elapsed,
            })

            logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"      F1-Macro: {metrics['f1_macro']:.4f}")
            logger.info(f"      Time: {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"      Failed: {e}")

    return results


# ============================================================
# Output helpers
# ============================================================

def save_comparison_plots(results_list, dataset_name, output_dir):
    """Save bar chart + confusion matrix plots.

    Parameters
    ----------
    results_list : list of dict
    dataset_name : str
    output_dir : Path
    """
    if not results_list:
        return

    bar_data = [{"method": r["method"], "accuracy": r["accuracy"],
                 "f1_macro": r["f1_macro"]} for r in results_list]
    fig, _ = plot_classifier_comparison(bar_data, title_prefix=f"{dataset_name} ")
    fig.savefig(output_dir / f"{dataset_name}_comparison.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_confusion_matrix_grid(results_list, title_prefix=f"{dataset_name} ")
    fig.savefig(output_dir / f"{dataset_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def write_csv_summary(output_dir, rows, filename="summary.csv"):
    """Write results to CSV.

    Parameters
    ----------
    output_dir : Path
    rows : list of dict
    filename : str
    """
    if not rows:
        return

    csv_path = output_dir / filename
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV saved: {csv_path}")


def init_experiment(title, output_subdir, config_path="configs/cwru.yaml"):
    """Initialize an experiment: style + config + output dir + logger.

    Parameters
    ----------
    title : str
    output_subdir : str
    config_path : str

    Returns
    -------
    config : ExperimentConfig
    output_dir : Path
    logger : Logger
    """
    setup_academic_style()
    config = ExperimentConfig.from_yaml(config_path)
    output_dir = Path(config.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)

    return config, output_dir, logger


# ============================================================
# Internal helpers
# ============================================================

def _extract_features_from_segments(signals_pre, fs_pre, label, seg_sec, overlap,
                                     all_features, all_labels):
    """Extract time+freq features from sliding window segments.

    Returns
    -------
    n_segments : int
    """
    seg_len = int(seg_sec * fs_pre)
    step = max(1, int(seg_len * (1.0 - overlap)))
    n_segments = max(0, (signals_pre.shape[1] - seg_len) // step + 1)

    for seg_idx in range(n_segments):
        start = seg_idx * step
        seg = signals_pre[0, start:start + seg_len]
        feat_td = extract_time_domain_features(seg)
        feat_fd = extract_freq_domain_features(seg, fs_pre)
        features = np.concatenate([feat_td, feat_fd])
        all_features.append(features)
        all_labels.append(label)

    return n_segments


# ============================================================
# BSS experiment helpers
# ============================================================

def build_tfa_config(config):
    """Build TFA config dict from ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    dict
    """
    cfg = {
        "mode": config.tfa.mode,
        "tfa_method": config.tfa.tfa_method,
        "n_bands": config.tfa.n_bands,
        "freq_range": config.tfa.freq_range,
        "wavelet": config.tfa.wavelet,
    }
    if hasattr(config.tfa, "bands_per_ch") and config.tfa.bands_per_ch is not None:
        cfg["bands_per_ch"] = config.tfa.bands_per_ch
    return cfg


def run_bss_experiment(config, signals_pre, fs_pre, output_dir, title_prefix=""):
    """Run a standard BSS experiment pipeline.

    Parameters
    ----------
    config : ExperimentConfig
    signals_pre : ndarray (n_channels, n_samples)
    fs_pre : float
    output_dir : Path
    title_prefix : str

    Returns
    -------
    dict
        {independence, ffds, n_sources}
    """
    from bss_test.tfa import build_observation_matrix
    from bss_test.bss import bss_factory
    from bss_test.evaluation import (
        compute_independence_metric,
        compute_fault_detection_score,
        plot_envelope_spectrum,
        plot_correlation_matrix,
    )

    tfa_config = build_tfa_config(config)
    X, labels = build_observation_matrix(signals_pre, fs_pre, tfa_config)
    logger.info(f"  Observation matrix: {X.shape[0]} obs × {X.shape[1]} samples")

    S_est, A_est, W = bss_factory(
        X, method=config.bss.method, n_components=config.bss.n_sources
    )
    logger.info(f"  Estimated sources: {S_est.shape[0]} × {S_est.shape[1]}")

    indep = compute_independence_metric(S_est)
    ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)
    logger.info(f"  Independence: {indep:.4f}, FFDS: {ffds:.2f}")

    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre, fault_freqs=config.feature_freqs,
        title_prefix=title_prefix,
    )
    fig.savefig(output_dir / "envelope_spectrum.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_correlation_matrix(S_est, title=f"{title_prefix}Source Correlation")
    fig.savefig(output_dir / "correlation_matrix.png", dpi=200)
    plt.close(fig)

    return {"independence": indep, "ffds": ffds, "n_sources": S_est.shape[0]}
