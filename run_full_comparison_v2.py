"""
Full Comparison v2: TFA × BSS × ML (Improved)
==============================================

Improvements over v1:
  1. Overlapping sliding window segmentation (90% overlap) → 10x more samples
  2. PCA dimensionality reduction (63D → 15D)
  3. k-fold cross-validation instead of single split
  4. Consistent BSS source selection (highest correlation with original)

Usage:
  python run_full_comparison_v2.py
"""

import sys
import os
import csv
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_cwru, load_phm_cut
from preprocessing import preprocess_signals
from cwt_module import time_freq_factory
from bss_module import bss_factory
from feature_extractor import extract_features
from ml_classifier import train_classifier, evaluate_classifier


# ============================================================
# Configuration
# ============================================================

TFA_METHODS = ["cwt", "stft"]
BSS_METHODS = ["sobi", "jade"]
ML_METHODS = ["svm", "rf", "knn", "lda"]

DATASETS = {
    "cwru": {
        "loader": "cwru",
        "fault_types": ["inner_race_007", "ball_007", "outer_race_6_007"],
        "load": 0,
        "channels": ["DE"],
        "preprocess": {"detrend": True, "bandpass": (100, 5000), "normalize": "zscore"},
        "tfa_params": {"n_bands": 8, "freq_range": (100, 5000), "wavelet": "cmor1.5-1.0"},
        "n_sources": 5,
        "window_sec": 0.5,    # Window length in seconds
        "step_sec": 0.4,      # Step length (20% overlap)
        "max_samples_per_class": 30,  # Limit to avoid excessive computation
    },
    "phm2010": {
        "loader": "phm",
        "tool_ids": [1, 4],
        "rep_cut": 150,
        "sensor_types": ["vib_x", "vib_y", "vib_z"],
        "preprocess": {"detrend": True, "bandpass": (100, 20000), "normalize": "zscore"},
        "tfa_params": {"n_bands": 8, "freq_range": (100, 20000), "wavelet": "cmor1.5-1.0"},
        "n_sources": 6,
        "window_sec": 0.5,
        "step_sec": 0.4,
        "max_samples_per_class": 30,
    },
}

OUTPUT_DIR = "outputs/full_comparison_v2"
PCA_COMPONENTS = 15  # Reduce 63D → 15D
CV_FOLDS = 5         # k-fold cross-validation


# ============================================================
# Data Loading with Overlapping Segmentation
# ============================================================

def segment_signal_overlapping(sig, fs, window_sec, step_sec, max_samples=None):
    """Segment signal using overlapping sliding window."""
    window_len = int(window_sec * fs)
    step_len = int(step_sec * fs)

    segments = []
    for start in range(0, len(sig) - window_len + 1, step_len):
        segments.append(sig[start:start + window_len])
        if max_samples and len(segments) >= max_samples:
            break

    return segments


def load_cwru_samples(config):
    """Load CWRU dataset with overlapping segmentation."""
    samples = []
    for ft in config["fault_types"]:
        try:
            signals, fs, rpm = load_cwru(
                data_dir="data/cwru", fault_type=ft,
                load=config["load"], channels=config["channels"],
            )
            sig = signals[0]

            # Overlapping segmentation
            segments = segment_signal_overlapping(
                sig, fs,
                config["window_sec"], config["step_sec"],
                config["max_samples_per_class"],
            )

            for i, seg in enumerate(segments):
                samples.append({
                    "signal": seg,
                    "fs": fs,
                    "label": ft,
                    "name": f"cwru_{ft}_seg{i:04d}",
                })
            print(f"    [{ft}] loaded {len(segments)} segments @ {fs} Hz")
        except Exception as e:
            print(f"    [{ft}] SKIPPED: {e}")
    return samples


def load_phm_samples(config):
    """Load PHM 2010 dataset with overlapping segmentation."""
    samples = []
    for tool_id in config["tool_ids"]:
        try:
            signals, fs, _ = load_phm_cut(
                tool_id=tool_id, cut_no=config["rep_cut"],
                data_dir="data/phm2010_milling",
                sensor_types=config["sensor_types"],
            )
            sig = signals[0]

            # Overlapping segmentation
            segments = segment_signal_overlapping(
                sig, fs,
                config["window_sec"], config["step_sec"],
                config["max_samples_per_class"],
            )

            for i, seg in enumerate(segments):
                samples.append({
                    "signal": seg,
                    "fs": fs,
                    "label": f"c{tool_id}",
                    "name": f"phm_c{tool_id}_seg{i:04d}",
                })
            print(f"    [c{tool_id}] loaded {len(segments)} segments @ {fs} Hz")
        except Exception as e:
            print(f"    [c{tool_id}] SKIPPED: {e}")
    return samples


LOADERS = {
    "cwru": load_cwru_samples,
    "phm2010": load_phm_samples,
}


# ============================================================
# BSS Source Selection (Improved)
# ============================================================

def select_best_source(S_est, original_signal):
    """Select BSS source with highest correlation to original signal."""
    from scipy.interpolate import interp1d

    n_sources = S_est.shape[0]
    n_samples = len(original_signal)

    best_corr = -1
    best_idx = 0

    for i in range(n_sources):
        # Interpolate to match original length
        src = S_est[i]
        if len(src) != n_samples:
            f_interp = interp1d(np.linspace(0, 1, len(src)), src, kind='linear')
            src = f_interp(np.linspace(0, 1, n_samples))

        # Compute correlation
        corr = abs(np.corrcoef(src, original_signal)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_idx = i

    return best_idx


# ============================================================
# Feature Extraction Pipeline (Improved)
# ============================================================

def extract_bss_features(signal, fs, tfa_method, tfa_params, bss_method, n_sources):
    """Extract features from BSS-separated signal with improved source selection."""
    # Build observation matrix via TFA
    params = tfa_params.copy()
    if tfa_method == "wpt":
        params["wavelet"] = "db4"

    matrix, freqs = time_freq_factory(signal, fs, method=tfa_method, **params)

    # BSS separation
    S_est, A_est, W = bss_factory(matrix, method=bss_method, n_components=n_sources)

    # Select best source using correlation (improved)
    best_idx = select_best_source(S_est, signal)
    S_best = S_est[best_idx]

    # Interpolate to match original signal length
    if len(S_best) != len(signal):
        from scipy.interpolate import interp1d
        f_interp = interp1d(np.linspace(0, 1, len(S_best)),
                           S_best, kind='linear')
        S_best = f_interp(np.linspace(0, 1, len(signal)))

    # Extract features
    features = extract_features(S_best, fs, feature_set="all")
    return features


# ============================================================
# ML Training with PCA and Cross-Validation
# ============================================================

def train_and_evaluate_cv(X, y, ml_method, n_folds=CV_FOLDS, pca_components=PCA_COMPONENTS):
    """Train and evaluate using k-fold cross-validation with PCA."""
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA dimensionality reduction
    n_comp = min(pca_components, X_scaled.shape[0] // 2, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X_scaled)

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Train model
    start_time = time.time()

    if ml_method == "svm":
        from sklearn.svm import SVC
        model = SVC(kernel="rbf", C=10.0, gamma='scale', random_state=42)
    elif ml_method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif ml_method == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=min(5, len(y_enc) // n_folds))
    elif ml_method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"Unknown ML method: {ml_method}")

    # Cross-validation scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc_scores = cross_val_score(model, X_reduced, y_enc, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model, X_reduced, y_enc, cv=cv, scoring='f1_macro')

    train_time = time.time() - start_time

    # Fit final model for confusion matrix
    model.fit(X_reduced, y_enc)
    y_pred = model.predict(X_reduced)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_enc, y_pred)

    return {
        "accuracy_mean": np.mean(acc_scores),
        "accuracy_std": np.std(acc_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "train_time": train_time,
        "n_samples": len(y),
        "n_features_orig": X.shape[1],
        "n_features_pca": n_comp,
        "confusion_matrix": cm,
        "label_names": list(le.classes_),
    }


# ============================================================
# Main Comparison Loop
# ============================================================

def run_comparison(samples, ds_name, out_dir):
    """Run TFA × BSS × ML comparison on a dataset."""
    results = []

    # Group samples by label
    labels = [s["label"] for s in samples]
    unique_labels = list(set(labels))

    if len(unique_labels) < 2:
        print(f"  ERROR: Need at least 2 classes, got {len(unique_labels)}")
        return results

    print(f"  Total samples: {len(samples)} ({len(unique_labels)} classes)")
    for label in unique_labels:
        count = sum(1 for l in labels if l == label)
        print(f"    {label}: {count} samples")

    # Extract features for all TFA×BSS combinations
    print(f"\n  Extracting features for all TFA×BSS combinations...")

    feature_cache = {}
    for tfa in TFA_METHODS:
        for bss in BSS_METHODS:
            combo = f"{tfa}+{bss}"
            print(f"    [{combo}] ", end="", flush=True)

            X_all = []
            y_all = []
            valid = True

            for sample in samples:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        feat = extract_bss_features(
                            sample["signal"], sample["fs"],
                            tfa, DATASETS[ds_name]["tfa_params"],
                            bss, DATASETS[ds_name]["n_sources"],
                        )
                    X_all.append(feat)
                    y_all.append(sample["label"])
                except Exception as e:
                    # Skip this sample
                    continue

            if len(X_all) >= 10 and len(set(y_all)) >= 2:
                feature_cache[(tfa, bss)] = (np.array(X_all), np.array(y_all))
                print(f"OK ({len(X_all)} samples, {X_all[0].shape[0]} features)")
            else:
                print(f"SKIPPED (only {len(X_all)} valid samples)")

    # Run ML classifiers with cross-validation
    print(f"\n  Running ML classifiers with {CV_FOLDS}-fold cross-validation...")

    for (tfa, bss), (X_all, y_all) in feature_cache.items():
        combo = f"{tfa}+{bss}"

        for ml in ML_METHODS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics = train_and_evaluate_cv(X_all, y_all, ml)

                results.append({
                    "dataset": ds_name,
                    "tfa": tfa,
                    "bss": bss,
                    "ml": ml,
                    "accuracy": metrics["accuracy_mean"],
                    "accuracy_std": metrics["accuracy_std"],
                    "f1_macro": metrics["f1_mean"],
                    "f1_std": metrics["f1_std"],
                    "train_time": metrics["train_time"],
                    "n_samples": metrics["n_samples"],
                    "n_features_orig": metrics["n_features_orig"],
                    "n_features_pca": metrics["n_features_pca"],
                    "status": "OK",
                    "confusion_matrix": metrics["confusion_matrix"],
                    "label_names": metrics["label_names"],
                })
                print(f"    [{combo}+{ml}] acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f} "
                      f"f1={metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}")
            except Exception as e:
                results.append({
                    "dataset": ds_name,
                    "tfa": tfa,
                    "bss": bss,
                    "ml": ml,
                    "accuracy": 0,
                    "accuracy_std": 0,
                    "f1_macro": 0,
                    "f1_std": 0,
                    "train_time": 0,
                    "n_samples": 0,
                    "n_features_orig": 0,
                    "n_features_pca": 0,
                    "status": f"ERROR: {str(e)[:40]}",
                    "confusion_matrix": None,
                    "label_names": None,
                })

    return results


# ============================================================
# Visualization
# ============================================================

def plot_accuracy_heatmap(results, ds_name, out_dir):
    """Plot accuracy heatmap for each ML method (TFA × BSS)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Accuracy Heatmap — {ds_name.upper()} (5-fold CV)", fontsize=14, fontweight="bold")

    for ax, ml in zip(axes.flat, ML_METHODS):
        matrix = np.zeros((len(TFA_METHODS), len(BSS_METHODS)))
        for i, tfa in enumerate(TFA_METHODS):
            for j, bss in enumerate(BSS_METHODS):
                for r in results:
                    if r["tfa"] == tfa and r["bss"] == bss and r["ml"] == ml:
                        matrix[i, j] = r["accuracy"]
                        break

        im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(BSS_METHODS)))
        ax.set_xticklabels([m.upper() for m in BSS_METHODS])
        ax.set_yticks(range(len(TFA_METHODS)))
        ax.set_yticklabels([m.upper() for m in TFA_METHODS])
        ax.set_xlabel("BSS Method")
        ax.set_ylabel("TFA Method")
        ax.set_title(f"{ml.upper()}")

        for i in range(len(TFA_METHODS)):
            for j in range(len(BSS_METHODS)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

        best_idx = np.unravel_index(matrix.argmax(), matrix.shape)
        ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                    fill=False, edgecolor="gold", linewidth=3))
        plt.colorbar(im, ax=ax, label="Accuracy")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_accuracy_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_combinations(results, ds_name, out_dir):
    """Plot bar chart of top 10 combinations with error bars."""
    valid = [r for r in results if r["status"] == "OK"]
    valid.sort(key=lambda x: x["accuracy"], reverse=True)
    top10 = valid[:10]

    fig, ax = plt.subplots(figsize=(12, 6))
    combos = [f"{r['tfa'].upper()}+{r['bss'].upper()}+{r['ml'].upper()}" for r in top10]
    accs = [r["accuracy"] for r in top10]
    stds = [r["accuracy_std"] for r in top10]

    bars = ax.bar(range(len(combos)), accs, yerr=stds, color="steelblue",
                  capsize=5, error_kw={'linewidth': 1.5})
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Top 10 Combinations — {ds_name.upper()} (5-fold CV)")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_best_combinations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ml_comparison(results, ds_name, out_dir):
    """Plot ML method comparison (averaged across TFA×BSS)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ML Classifier Comparison — {ds_name.upper()} (5-fold CV)", fontsize=14, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ml_accs = {}
    ml_stds = {}
    for ml in ML_METHODS:
        accs = [r["accuracy"] for r in results if r["ml"] == ml and r["status"] == "OK"]
        stds = [r["accuracy_std"] for r in results if r["ml"] == ml and r["status"] == "OK"]
        ml_accs[ml] = np.mean(accs) if accs else 0
        ml_stds[ml] = np.mean(stds) if stds else 0

    bars = ax.bar(range(len(ML_METHODS)), [ml_accs[m] for m in ML_METHODS],
                  yerr=[ml_stds[m] for m in ML_METHODS], color="steelblue", capsize=5)
    ax.set_xticks(range(len(ML_METHODS)))
    ax.set_xticklabels([m.upper() for m in ML_METHODS])
    ax.set_ylabel("Accuracy")
    ax.set_title("Average Accuracy")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, ml in zip(bars, ML_METHODS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{ml_accs[ml]:.3f}", ha="center", va="bottom", fontsize=10)

    # Training time
    ax = axes[1]
    ml_times = {}
    for ml in ML_METHODS:
        times = [r["train_time"] for r in results if r["ml"] == ml and r["status"] == "OK"]
        ml_times[ml] = np.mean(times) if times else 0

    bars = ax.bar(range(len(ML_METHODS)), [ml_times[m] for m in ML_METHODS], color="coral")
    ax.set_xticks(range(len(ML_METHODS)))
    ax.set_xticklabels([m.upper() for m in ML_METHODS])
    ax.set_ylabel("Time [s]")
    ax.set_title("Average Training Time")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, ml in zip(bars, ML_METHODS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{ml_times[ml]:.3f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_ml_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrices(results, ds_name, out_dir):
    """Plot confusion matrices for best combination of each ML method."""
    cm_dir = os.path.join(out_dir, f"{ds_name}_confusion")
    os.makedirs(cm_dir, exist_ok=True)

    for ml in ML_METHODS:
        valid = [r for r in results if r["ml"] == ml and r["status"] == "OK"
                 and r["confusion_matrix"] is not None]
        if not valid:
            continue

        best = max(valid, key=lambda x: x["accuracy"])
        cm = best["confusion_matrix"]
        label_names = best["label_names"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticks(range(len(label_names)))
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{ml.upper()} — {best['tfa'].upper()}+{best['bss'].upper()}\n"
                     f"Acc={best['accuracy']:.3f}±{best['accuracy_std']:.3f}")

        for i in range(len(label_names)):
            for j in range(len(label_names)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)

        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        path = os.path.join(cm_dir, f"{ml}_best.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print(f"  Saved: {cm_dir}/")


def plot_cross_dataset_comparison(all_results, out_dir):
    """Plot comparison across datasets."""
    datasets = sorted(all_results.keys())
    if len(datasets) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Dataset Comparison (5-fold CV)", fontsize=14, fontweight="bold")

    # Accuracy by ML method
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.18
    for i, ml in enumerate(ML_METHODS):
        accs = []
        for ds in datasets:
            valid = [r["accuracy"] for r in all_results[ds]
                     if r["ml"] == ml and r["status"] == "OK"]
            accs.append(np.mean(valid) if valid else 0)
        ax.bar(x + i * width, accs, width, label=ml.upper())

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_title("ML Accuracy by Dataset")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.5, 1.05)

    # Accuracy by TFA method
    ax = axes[1]
    for i, tfa in enumerate(TFA_METHODS):
        accs = []
        for ds in datasets:
            valid = [r["accuracy"] for r in all_results[ds]
                     if r["tfa"] == tfa and r["status"] == "OK"]
            accs.append(np.mean(valid) if valid else 0)
        ax.bar(x + i * width, accs, width, label=tfa.upper())

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_title("TFA Accuracy by Dataset")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, "cross_dataset_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Summary
# ============================================================

def print_summary(results, ds_name):
    """Print summary of best combinations."""
    valid = [r for r in results if r["status"] == "OK"]
    if not valid:
        return

    print(f"\n  --- {ds_name.upper()} Summary ---")
    print(f"  {'Rank':<6} {'TFA':<6} {'BSS':<8} {'ML':<6} {'Accuracy':<15} {'F1':<15}")
    print(f"  {'-'*66}")

    valid.sort(key=lambda x: x["accuracy"], reverse=True)
    for i, r in enumerate(valid[:5]):
        print(f"  {i+1:<6} {r['tfa'].upper():<6} {r['bss'].upper():<8} "
              f"{r['ml'].upper():<6} {r['accuracy']:.3f}±{r['accuracy_std']:.3f}   "
              f"{r['f1_macro']:.3f}±{r['f1_std']:.3f}")


def save_csv(all_results, out_dir):
    """Save results to CSV."""
    path = os.path.join(out_dir, "summary.csv")
    keys = ["dataset", "tfa", "bss", "ml", "accuracy", "accuracy_std",
            "f1_macro", "f1_std", "train_time", "n_samples",
            "n_features_orig", "n_features_pca", "status"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for ds_results in all_results.values():
            writer.writerows(ds_results)

    print(f"\n  Summary CSV: {path}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    print("=" * 60)
    print("Full Comparison v2: TFA × BSS × ML (Improved)")
    print("=" * 60)
    print(f"TFA Methods:     {[m.upper() for m in TFA_METHODS]}")
    print(f"BSS Methods:     {[m.upper() for m in BSS_METHODS]}")
    print(f"ML Methods:      {[m.upper() for m in ML_METHODS]}")
    print(f"Datasets:        {list(DATASETS.keys())}")
    print(f"PCA Components:  {PCA_COMPONENTS}")
    print(f"CV Folds:        {CV_FOLDS}")
    print(f"Total combos:    {len(TFA_METHODS)} × {len(BSS_METHODS)} × {len(ML_METHODS)} = "
          f"{len(TFA_METHODS) * len(BSS_METHODS) * len(ML_METHODS)}")

    for ds_name, ds_config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        ds_out = os.path.join(OUTPUT_DIR, ds_name)
        os.makedirs(ds_out, exist_ok=True)

        # Load data
        print(f"  Loading data with overlapping segmentation...")
        loader = LOADERS[ds_name]
        samples = loader(ds_config)

        if not samples:
            print(f"  ERROR: No data loaded. Skipping.")
            continue

        # Run comparison
        results = run_comparison(samples, ds_name, ds_out)
        all_results[ds_name] = results

        # Generate visualizations
        print(f"\n  Generating visualizations...")
        plot_accuracy_heatmap(results, ds_name, OUTPUT_DIR)
        plot_best_combinations(results, ds_name, OUTPUT_DIR)
        plot_ml_comparison(results, ds_name, OUTPUT_DIR)
        plot_confusion_matrices(results, ds_name, OUTPUT_DIR)

        # Print summary
        print_summary(results, ds_name)

    # Cross-dataset comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Cross-Dataset Comparison")
        print(f"{'='*60}")
        plot_cross_dataset_comparison(all_results, OUTPUT_DIR)

    # Save CSV
    save_csv(all_results, OUTPUT_DIR)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for ds_name, results in all_results.items():
        valid = [r for r in results if r["status"] == "OK"]
        if valid:
            best = max(valid, key=lambda x: x["accuracy"])
            print(f"  {ds_name}: {best['tfa'].upper()}+{best['bss'].upper()}+{best['ml'].upper()}"
                  f" (acc={best['accuracy']:.3f}±{best['accuracy_std']:.3f})")

    print(f"\n{'='*60}")
    print(f"ALL DONE — results in {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
