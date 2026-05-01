"""
Evaluation and visualization for BSS results.

Includes:
- Time-domain waveform comparison (mixed vs separated)
- Spectrum comparison (FFT)
- Envelope spectrum (Hilbert + FFT) for bearing fault detection
- Wear evolution curves (PHM 2010 specific)
- Quantitative metrics: SIR, SDR, correlation (synthetic mixtures only)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, welch


# ============================================================
# Academic paper style configuration
# ============================================================

ACADEMIC_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
}


def setup_academic_style():
    """Apply academic paper style to all subsequent plots."""
    plt.rcParams.update(ACADEMIC_STYLE)


def evaluate_bss(S_est, W, X_original, fs, config=None):
    """
    Main evaluation entry point.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)
        Estimated sources.
    W : ndarray (n_sources, n_obs)
        Demixing matrix.
    X_original : ndarray (n_obs, n_samples)
        Original observation (mixed) signals.
    fs : float
        Sampling rate.
    config : dict
        Additional configuration.
    """
    print(f"\n{'='*60}")
    print(f"BSS Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Observations: {X_original.shape[0]}, Sources: {S_est.shape[0]}")
    print(f"  Samples: {S_est.shape[1]}, Sampling rate: {fs} Hz")
    print(f"  Duration: {S_est.shape[1]/fs:.2f} s")
    print(f"{'='*60}\n")


def plot_waveform_comparison(X_original, S_est, fs, n_show=6, title_prefix="",
                              max_duration=None):
    """
    Plot time-domain waveforms: observations vs separated sources.

    Parameters
    ----------
    X_original : ndarray (n_obs, n_samples)
    S_est : ndarray (n_sources, n_samples)
    fs : float
    n_show : int
        Max number of signals to show in each panel.
    title_prefix : str
    max_duration : float or None
        Max time to display in seconds.
    """
    setup_academic_style()
    n_obs = X_original.shape[0]
    n_src = S_est.shape[0]
    n_samples = X_original.shape[1]

    time_axis = np.arange(n_samples) / fs
    if max_duration is not None:
        cutoff = int(max_duration * fs)
        time_axis = time_axis[:cutoff]
        X_obs_plot = X_original[:, :cutoff]
        S_plot = S_est[:, :cutoff]
    else:
        X_obs_plot = X_original
        S_plot = S_est

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Observations
    ax = axes[0]
    n_plot = min(n_obs, n_show)
    # Adaptive offset: 1.5 * max amplitude to prevent overlap
    max_amp = max(np.max(np.abs(X_obs_plot[i])) for i in range(n_plot))
    offsets = np.arange(n_plot) * 1.5 * max_amp
    for i in range(n_plot):
        ax.plot(time_axis, X_obs_plot[i] + offsets[i], linewidth=0.5)
    ax.set_title(f"{title_prefix}Observations (Mixed Signals)")
    ax.set_ylabel("Amplitude (offset)")
    ax.legend([f"Obs {i}" for i in range(n_plot)], loc="upper right", fontsize=8)

    # Separated sources
    ax = axes[1]
    n_plot = min(n_src, n_show)
    max_amp = max(np.max(np.abs(S_plot[i])) for i in range(n_plot))
    offsets = np.arange(n_plot) * 1.5 * max_amp
    for i in range(n_plot):
        ax.plot(time_axis, S_plot[i] + offsets[i], linewidth=0.5)
    ax.set_title(f"{title_prefix}Separated Sources")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude (offset)")
    ax.legend([f"Source {i}" for i in range(n_plot)], loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig, axes


def plot_spectrum_comparison(X_original, S_est, fs, n_show=5, title_prefix=""):
    """
    Plot FFT magnitude spectra of observations vs separated sources.

    Parameters
    ----------
    X_original : ndarray (n_obs, n_samples)
    S_est : ndarray (n_sources, n_samples)
    fs : float
    n_show : int
    title_prefix : str
    """
    setup_academic_style()
    n_obs = X_original.shape[0]
    n_src = S_est.shape[0]
    N = X_original.shape[1]

    freq_axis = np.fft.rfftfreq(N, 1.0 / fs)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    n_plot = min(n_obs, n_show)
    for i in range(n_plot):
        spectrum = np.abs(np.fft.rfft(X_original[i]))
        spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        ax.plot(freq_axis, spectrum + i * 1.2, linewidth=0.5, label=f"Obs {i}")
    ax.set_title(f"{title_prefix}Observation Spectra")
    ax.set_ylabel("Normalized Magnitude (offset)")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    n_plot = min(n_src, n_show)
    for i in range(n_plot):
        spectrum = np.abs(np.fft.rfft(S_est[i]))
        spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        ax.plot(freq_axis, spectrum + i * 1.2, linewidth=0.5, label=f"Source {i}")
    ax.set_title(f"{title_prefix}Separated Source Spectra")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized Magnitude (offset)")
    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    return fig, axes


def plot_envelope_spectrum(signals, fs, fault_freqs=None, n_show=6,
                            labels=None, title_prefix=""):
    """
    Plot envelope spectrum (Hilbert envelope -> FFT) for fault frequency detection.

    Parameters
    ----------
    signals : ndarray (n_signals, n_samples)
        Signals to analyze (typically separated sources).
    fs : float
    fault_freqs : dict or None
        Dict of {label: frequency} for fault characteristic frequencies.
        E.g. {"BPFO": 162.2, "BPFI": 243.6, "BSF": 109.8}
    n_show : int
    labels : list or None
    title_prefix : str
    """
    setup_academic_style()
    n_sig = min(signals.shape[0], n_show)
    fig, axes = plt.subplots(n_sig, 1, figsize=(12, 2.5 * n_sig), sharex=True)
    if n_sig == 1:
        axes = [axes]

    # Adaptive x-axis range
    if fault_freqs:
        max_fault_freq = max(fault_freqs.values())
        x_max = min(max_fault_freq * 2.5, fs / 2)
    else:
        x_max = min(fs / 2, 1000)

    for i in range(n_sig):
        ax = axes[i]
        sig = signals[i]
        # Hilbert envelope
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        # FFT of envelope
        N = len(envelope)
        env_spectrum = np.abs(np.fft.rfft(envelope))
        freq = np.fft.rfftfreq(N, 1.0 / fs)

        ax.plot(freq, env_spectrum, color="#1976D2")
        ax.set_xlim([0, x_max])

        # Adaptive y-axis: use 98th percentile to avoid outlier compression
        freq_mask = freq <= x_max
        if np.any(freq_mask):
            y_max = np.percentile(env_spectrum[freq_mask][1:], 98) * 1.2
            ax.set_ylim([0, y_max])

        label = labels[i] if labels else f"Source {i}"
        ax.set_title(f"{title_prefix}{label}")
        ax.set_ylabel("Amplitude")

        # Mark fault frequencies with legend (not text on plot)
        if fault_freqs is not None:
            for j, (fname, ff) in enumerate(fault_freqs.items()):
                ax.axvline(x=ff, color="#D32F2F", linestyle="--", alpha=0.7,
                           linewidth=1.0, label=fname if i == 0 else "")
                # Harmonics
                for h in [2, 3]:
                    ax.axvline(x=h * ff, color="#D32F2F", linestyle=":",
                               alpha=0.25, linewidth=0.6)
            if i == 0:
                ax.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")

    axes[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    return fig, axes


def plot_correlation_matrix(S_est, title="Source Correlation Matrix"):
    """
    Plot correlation matrix between separated sources.
    Ideally should be close to diagonal.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)
    title : str
    """
    setup_academic_style()
    n_src = S_est.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))

    if n_src == 1:
        corr = np.array([[1.0]])
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels(["S0"])
        ax.set_yticklabels(["S0"])
        ax.text(0, 0, "1.00", ha="center", va="center", fontsize=14, fontweight="bold")
    else:
        corr = np.corrcoef(S_est)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_src))
        ax.set_yticks(range(n_src))
        ax.set_xticklabels([f"S{i}" for i in range(n_src)])
        ax.set_yticklabels([f"S{i}" for i in range(n_src)])

        # Annotate cells
        for i in range(n_src):
            for j in range(n_src):
                val = corr[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    return fig, ax


def plot_wear_evolution(S_list, wear_labels, tool_id=1, title_prefix=""):
    """
    PHM 2010 specific: plot source energy vs tool wear progression.

    Parameters
    ----------
    S_list : list of ndarray
        Each element is S_est for one cut: shape (n_sources, n_samples).
    wear_labels : ndarray (n_cuts,)
        Wear values (or cut numbers) for x-axis.
    tool_id : int
    title_prefix : str

    Returns
    -------
    fig, ax
    """
    setup_academic_style()
    n_cuts = len(S_list)
    n_sources = S_list[0].shape[0]

    # Compute energy (RMS) for each source in each cut
    energy = np.zeros((n_cuts, n_sources))
    for k in range(n_cuts):
        for s in range(n_sources):
            energy[k, s] = np.sqrt(np.mean(S_list[k][s] ** 2))

    fig, axes = plt.subplots(n_sources, 1, figsize=(10, 2 * n_sources),
                              sharex=True)
    if n_sources == 1:
        axes = [axes]

    x_axis = np.arange(n_cuts) + 1
    colors = plt.cm.Set2(np.linspace(0, 1, n_sources))
    for s in range(n_sources):
        ax = axes[s]
        ax.plot(x_axis, energy[:, s], "o-", markersize=4, linewidth=1.2,
                color=colors[s], label=f"Source {s} energy")
        ax.set_ylabel("RMS Energy")
        ax.set_title(f"{title_prefix}Source {s}")
        ax.legend(loc="upper left", fontsize=8)

        # Twin axis for wear
        ax2 = ax.twinx()
        ax2.plot(x_axis[:len(wear_labels)], wear_labels[:n_cuts],
                 "r--", linewidth=1.0, label="Wear", alpha=0.7)
        ax2.set_ylabel("Wear [10^-3 mm]", color="#D32F2F")
        ax2.tick_params(axis="y", labelcolor="#D32F2F")
        ax2.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Cut Number")
    plt.tight_layout()
    return fig, axes


def plot_bss_metrics_comparison(results_dict, title_prefix=""):
    """
    Compare BSS methods via grouped bar chart (FFDS + Independence).

    Parameters
    ----------
    results_dict : dict
        {method_name: {"independence": float, "ffds": float}}
    title_prefix : str

    Returns
    -------
    fig, (ax1, ax2)
    """
    setup_academic_style()
    methods = list(results_dict.keys())
    ffds_vals = [results_dict[m]["ffds"] for m in methods]
    indep_vals = [results_dict[m]["independence"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, len(methods) * 1.8), 5))

    bars_ffds = ax1.bar(x - width / 2, ffds_vals, width, label="FFDS (higher=better)",
                        color="#1976D2", alpha=0.85)
    ax1.set_ylabel("FFDS", color="#1976D2")
    ax1.tick_params(axis="y", labelcolor="#1976D2")

    ax2 = ax1.twinx()
    bars_indep = ax2.bar(x + width / 2, indep_vals, width,
                         label="Independence (lower=better)",
                         color="#FF9800", alpha=0.85)
    ax2.set_ylabel("Independence", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    ax1.set_xlabel("BSS Method")
    ax1.set_title(f"{title_prefix}BSS Method Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in methods])

    for bar in bars_ffds:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     fontsize=8, color="#1976D2")

    for bar in bars_indep:
        height = bar.get_height()
        ax2.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     fontsize=8, color="#FF9800")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_tfa_metrics_comparison(results_dict, title_prefix=""):
    """
    Compare TFA methods via bar chart (FFDS).

    Parameters
    ----------
    results_dict : dict
        {method_name: {"ffds": float, ...}}
    title_prefix : str

    Returns
    -------
    fig, ax
    """
    setup_academic_style()
    methods = list(results_dict.keys())
    ffds_vals = [results_dict[m]["ffds"] for m in methods]

    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C62828", "#00796B"]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 5))
    bars = ax.bar(methods, ffds_vals, color=bar_colors, alpha=0.85, width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("TFA Method")
    ax.set_ylabel("FFDS (higher = better)")
    ax.set_title(f"{title_prefix}TFA Method Comparison — Fault Detection Score")
    ax.set_ylim(0, max(ffds_vals) * 1.2 if max(ffds_vals) > 0 else 1)

    plt.tight_layout()
    return fig, ax


def compute_metrics(S_true, S_est):
    """
    Compute SIR and correlation for synthetic mixture evaluation.

    Parameters
    ----------
    S_true : ndarray (n_sources, n_samples)
        Ground truth sources.
    S_est : ndarray (n_sources, n_samples)
        Estimated sources.

    Returns
    -------
    metrics : dict
        {"SIR": ..., "correlation": ...}
    """
    n_src = min(S_true.shape[0], S_est.shape[0])
    # Normalize both
    S_true_norm = S_true[:n_src] / (np.std(S_true[:n_src], axis=1, keepdims=True) + 1e-12)
    S_est_norm = S_est[:n_src] / (np.std(S_est[:n_src], axis=1, keepdims=True) + 1e-12)

    # Find best permutation via correlation
    corr_matrix = np.abs(np.corrcoef(S_true_norm, S_est_norm)[:n_src, n_src:])
    # Greedy matching
    best_corrs = []
    used = set()
    for _ in range(n_src):
        best_val = -1
        best_pair = (-1, -1)
        for ii in range(corr_matrix.shape[0]):
            for jj in range(corr_matrix.shape[1]):
                if jj not in used and corr_matrix[ii, jj] > best_val:
                    best_val = corr_matrix[ii, jj]
                    best_pair = (jj,)
        if best_val > -1:
            used.add(best_pair[0])
            best_corrs.append(best_val)

    mean_corr = np.mean(best_corrs) if best_corrs else 0.0

    # Simple SIR approximation
    errors = []
    for i in range(n_src):
        source_energy = np.var(S_true_norm[i])
        residual = S_true_norm[i] - S_est_norm[i]
        noise_energy = np.var(residual)
        if noise_energy > 1e-12:
            errors.append(10 * np.log10(source_energy / noise_energy))
    mean_sir = np.mean(errors) if errors else float("inf")

    return {"SIR_dB": mean_sir, "mean_correlation": mean_corr}


# ============================================================
# Comparison visualization functions
# ============================================================


def plot_tfa_comparison(signals_dict, original_signal, fs, title_prefix=""):
    """
    Compare TFA method reconstructions vs original signal in time domain.

    Parameters
    ----------
    signals_dict : dict
        {method_name: ndarray (n_bands, n_samples)} — TFA outputs.
    original_signal : ndarray (n_samples,)
        Original 1D signal for reference.
    fs : float
        Sampling rate.
    title_prefix : str

    Returns
    -------
    fig, axes
    """
    setup_academic_style()
    n_methods = len(signals_dict)
    n_show_bands = 3  # Show first 3 bands per method

    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(12, 2.5 * (n_methods + 1)),
                              sharex=True)

    # Original signal
    t = np.arange(len(original_signal)) / fs
    max_dur = min(0.5, t[-1])
    n_plot = int(max_dur * fs)
    t_plot = t[:n_plot]

    axes[0].plot(t_plot, original_signal[:n_plot], "k", linewidth=0.6)
    axes[0].set_ylabel("Original")
    axes[0].set_title(f"{title_prefix}Original Signal")

    method_names = list(signals_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    for idx, method_name in enumerate(method_names):
        ax = axes[idx + 1]
        matrix = signals_dict[method_name]
        n_bands = min(matrix.shape[0], n_show_bands)
        max_amp = max(np.max(np.abs(matrix[b][:n_plot])) for b in range(n_bands))
        offsets = np.arange(n_bands) * 1.5 * max_amp
        for b in range(n_bands):
            ax.plot(t_plot, matrix[b][:n_plot] + offsets[b], linewidth=0.5,
                    label=f"band {b}", color=plt.cm.tab10(b))
        ax.set_ylabel(method_name.upper())
        ax.set_title(f"{title_prefix}TFA: {method_name.upper()} (top {n_bands} bands)")
        if n_bands <= 5:
            ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    return fig, axes


def plot_bss_comparison(results_dict, fs, fault_freqs=None, n_sources_show=4,
                         title_prefix=""):
    """
    Compare BSS algorithm results via envelope spectra subplot grid.

    Parameters
    ----------
    results_dict : dict
        {method_name: ndarray (n_sources, n_samples)} — separated sources.
    fs : float
        Sampling rate.
    fault_freqs : dict or None
        Fault frequency markers.
    n_sources_show : int
        Number of source columns (from each method) to show.
    title_prefix : str

    Returns
    -------
    fig, axes
    """
    setup_academic_style()
    n_methods = len(results_dict)
    method_names = list(results_dict.keys())

    fig, axes = plt.subplots(n_methods, n_sources_show,
                              figsize=(4 * n_sources_show, 2.5 * n_methods),
                              sharex=True)
    if n_methods == 1 and n_sources_show == 1:
        axes = np.array([[axes]])
    elif n_methods == 1:
        axes = np.array([axes])
    elif n_sources_show == 1:
        axes = axes[:, np.newaxis]

    # Adaptive x-axis range
    if fault_freqs:
        max_fault_freq = max(fault_freqs.values())
        x_max = min(max_fault_freq * 2.5, fs / 2)
    else:
        x_max = min(fs / 2, 1000)

    for i, method_name in enumerate(method_names):
        S_est = results_dict[method_name]
        n_src = min(S_est.shape[0], n_sources_show)

        # Compute y_max for this row (shared y-axis)
        row_y_max = 0
        for j in range(n_src):
            sig = S_est[j]
            analytic = hilbert(sig)
            envelope = np.abs(analytic)
            N = len(envelope)
            env_spec = np.abs(np.fft.rfft(envelope))
            freq = np.fft.rfftfreq(N, 1.0 / fs)
            freq_mask = freq <= x_max
            if np.any(freq_mask):
                row_y_max = max(row_y_max, np.percentile(env_spec[freq_mask][1:], 98))

        for j in range(n_sources_show):
            ax = axes[i, j]
            if j < n_src:
                sig = S_est[j]
                analytic = hilbert(sig)
                envelope = np.abs(analytic)
                N = len(envelope)
                env_spec = np.abs(np.fft.rfft(envelope))
                freq = np.fft.rfftfreq(N, 1.0 / fs)
                ax.plot(freq, env_spec, linewidth=0.6, color="#1976D2")
                ax.set_xlim([0, x_max])
                ax.set_ylim([0, row_y_max * 1.2])
                if fault_freqs is not None:
                    for fname, ff in fault_freqs.items():
                        ax.axvline(x=ff, color="#D32F2F", linestyle="--", alpha=0.4,
                                   linewidth=0.6)
                ax.set_title(f"{method_name.upper()} S{j}", fontsize=9)
            else:
                ax.set_visible(False)
        if i == 0:
            for j in range(n_sources_show):
                axes[i, j].set_title(f"S{j}", fontsize=9)

    for j in range(n_sources_show):
        axes[-1, j].set_xlabel("Freq [Hz]")
    for i in range(n_methods):
        axes[i, 0].set_ylabel(method_names[i].upper() + "\nAmplitude", fontsize=8)

    fig.suptitle(f"{title_prefix}BSS Method Comparison — Envelope Spectra",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig, axes


def plot_classifier_comparison(results_list, title_prefix=""):
    """
    Compare ML classifiers via grouped bar chart (accuracy + F1).

    Parameters
    ----------
    results_list : list of dict
        Each dict: {"method": str, "accuracy": float, "f1_macro": float}
    title_prefix : str

    Returns
    -------
    fig, ax
    """
    setup_academic_style()
    methods = [r["method"] for r in results_list]
    accuracies = [r["accuracy"] * 100 for r in results_list]
    f1_scores = [r["f1_macro"] * 100 for r in results_list]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)",
                   color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Macro (%)",
                   color="#FF9800", alpha=0.85)

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{title_prefix}Classifier Comparison — Accuracy & F1-Macro")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)

    # Annotate bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8)

    plt.tight_layout()
    return fig, ax


def plot_confusion_matrix_grid(results_list, title_prefix=""):
    """
    Plot confusion matrices for all classifiers in a subplot grid.

    Parameters
    ----------
    results_list : list of dict
        Each dict:
          {"method": str, "confusion_matrix": ndarray (n_classes, n_classes),
           "label_names": list or None}
    title_prefix : str

    Returns
    -------
    fig, axes
    """
    setup_academic_style()
    n_methods = len(results_list)
    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, result in enumerate(results_list):
        ax = axes_flat[idx]
        method = result["method"]
        cm = result["confusion_matrix"]

        im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
        n_cls = cm.shape[0]
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))

        label_names = result.get("label_names")
        if label_names is not None and n_cls <= 8:
            ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(label_names, fontsize=8)
        else:
            ax.set_xticklabels(range(n_cls))
            ax.set_yticklabels(range(n_cls))

        ax.set_title(f"{method.upper()}", fontsize=11)
        if idx % n_cols == 0:
            ax.set_ylabel("True")
        if idx >= n_methods - n_cols:
            ax.set_xlabel("Predicted")

        # Annotate cells
        for i in range(n_cls):
            for j in range(n_cls):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"{title_prefix}Confusion Matrix Comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig, axes


def plot_tfa_bss_cross_comparison(cross_results, fs, fault_freqs=None,
                                   title_prefix=""):
    """
    TFA × BSS cross-comparison grid: each cell shows envelope spectrum of
    the first separated source.

    Parameters
    ----------
    cross_results : dict
        {(tfa_method, bss_method): ndarray (n_sources, n_samples)}
    fs : float
    fault_freqs : dict or None
        Fault frequency markers, e.g. {"BPFO": 107.3, "BPFI": 162.2, "BSF": 70.6}
    title_prefix : str

    Returns
    -------
    fig, axes
    """
    setup_academic_style()
    tfa_methods = sorted(set(k[0] for k in cross_results.keys()))
    bss_methods = sorted(set(k[1] for k in cross_results.keys()))
    n_rows = len(tfa_methods)
    n_cols = len(bss_methods)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.2 * n_cols, 3.2 * n_rows),
                              sharex=True, sharey=False,
                              squeeze=False)

    # Adaptive x-axis range
    if fault_freqs:
        max_fault_freq = max(fault_freqs.values())
        x_max = min(max_fault_freq * 2.5, fs / 2)
    else:
        x_max = min(fs / 2, 1000)

    for i, tfa in enumerate(tfa_methods):
        for j, bss in enumerate(bss_methods):
            ax = axes[i, j]
            key = (tfa, bss)
            if key in cross_results:
                S_est = cross_results[key]
                sig = S_est[0]
                analytic = hilbert(sig)
                envelope = np.abs(analytic)
                N = len(envelope)
                env_spec = np.abs(np.fft.rfft(envelope))
                freq = np.fft.rfftfreq(N, 1.0 / fs)
                ax.plot(freq, env_spec, linewidth=0.6, color="#1976D2")
                ax.set_xlim([0, x_max])

                # Adaptive y-axis
                freq_mask = freq <= x_max
                if np.any(freq_mask):
                    y_max = np.percentile(env_spec[freq_mask][1:], 98) * 1.2
                    ax.set_ylim([0, y_max])

                # Fault frequency markers
                if fault_freqs is not None:
                    for fname, ff in fault_freqs.items():
                        ax.axvline(x=ff, color="#D32F2F", linestyle="--", alpha=0.5,
                                   linewidth=0.7)
                        for h in [2, 3]:
                            ax.axvline(x=h * ff, color="#D32F2F", linestyle=":",
                                       alpha=0.2, linewidth=0.5)

                # Source independence value (top-right annotation)
                ind = compute_independence_metric(S_est)
                ax.text(0.98, 0.92, f"ind={ind:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.8))

                ax.set_ylabel("Amp", fontsize=7)
                ax.tick_params(labelsize=6)
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

            # Column titles (BSS method)
            if i == 0:
                ax.set_title(bss.upper(), fontsize=10, fontweight="bold",
                             color="#333333")
            # Row labels (TFA method)
            if j == 0:
                ax.set_ylabel(tfa.upper() + "\nAmp", fontsize=8, fontweight="bold")

        # Last row: x-label
        axes[n_rows - 1, j].set_xlabel("Freq [Hz]", fontsize=8)

    fig.suptitle(f"{title_prefix}TFA × BSS Cross-Comparison — Envelope Spectra",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, axes


def compute_independence_metric(S_est):
    """
    Compute source independence: mean absolute off-diagonal correlation.

    Lower = sources are more independent = better separation.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)

    Returns
    -------
    float : mean |off-diagonal correlation|, 0 = perfect independence
    """
    n_src = S_est.shape[0]
    if n_src < 2:
        return 0.0
    corr = np.corrcoef(S_est)
    mask = ~np.eye(n_src, dtype=bool)
    off_diag = np.abs(corr[mask])
    return float(np.mean(off_diag))


def compute_fault_detection_score(S_est, fs, fault_freqs, tol_hz=5.0):
    """
    Fault Frequency Detection Score (FFDS):
    mean(peak at each fault freq) / median noise floor of envelope spectrum.

    Higher = fault frequency more prominent = better diagnostic value.

    Parameters
    ----------
    S_est : ndarray (n_sources, n_samples)
    fs : float
    fault_freqs : dict {name: freq_hz}
    tol_hz : float

    Returns
    -------
    float : FFDS score
    """
    # Use first source
    sig = S_est[0]
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    N = len(envelope)
    env_spec = np.abs(np.fft.rfft(envelope))
    freq = np.fft.rfftfreq(N, 1.0 / fs)

    # Noise floor: median over 20–500 Hz (excluding DC)
    mask_noise = (freq >= 20) & (freq <= 500)
    if np.sum(mask_noise) < 10:
        return 0.0
    noise_floor = np.median(env_spec[mask_noise]) + 1e-12

    # Peak values at fault frequencies
    peaks = []
    for ff in fault_freqs.values():
        mask = (freq >= ff - tol_hz) & (freq <= ff + tol_hz)
        if np.sum(mask) > 0:
            peaks.append(float(np.max(env_spec[mask])))
    if not peaks:
        return 0.0

    return float(np.mean(peaks)) / noise_floor


def plot_separation_quality_report(cross_results, fs, fault_freqs=None,
                                    title_prefix=""):
    """
    Two-metric comprehensive separation quality report.

    Layout:
      Upper:  4x4 envelope spectrum grid (compact)
      Lower:  Two side-by-side heatmaps
        Left  — Source Independence (mean |off-diag corr|, lower = better)
        Right — Fault Detection Score FFDS (peak/noise ratio, higher = better)
      Best combination auto-highlighted (gold border + BEST label).

    Parameters
    ----------
    cross_results : dict
        {(tfa, bss): S_est ndarray (n_sources, n_samples)}
    fs : float
    fault_freqs : dict or None
    title_prefix : str

    Returns
    -------
    fig : matplotlib Figure
    metrics_grid : dict
        {(tfa, bss): {"independence": float, "ffds": float}}
    """
    setup_academic_style()
    from matplotlib.patches import FancyBboxPatch

    tfa_methods = sorted(set(k[0] for k in cross_results.keys()))
    bss_methods = sorted(set(k[1] for k in cross_results.keys()))
    n_rows = len(tfa_methods)
    n_cols = len(bss_methods)

    # --- Compute metrics ---
    metrics_grid = {}
    for i, tfa in enumerate(tfa_methods):
        for j, bss in enumerate(bss_methods):
            key = (tfa, bss)
            if key in cross_results:
                S_est = cross_results[key]
                indep = compute_independence_metric(S_est)
                ffds = compute_fault_detection_score(S_est, fs, fault_freqs) \
                    if fault_freqs else 0.0
                metrics_grid[key] = {"independence": indep, "ffds": ffds}
            else:
                metrics_grid[key] = {"independence": np.nan, "ffds": np.nan}

    # Find best combination (by FFDS, then independence)
    valid_keys = [k for k, v in metrics_grid.items()
                  if not np.isnan(v["ffds"]) and v["ffds"] > 0]
    if valid_keys:
        best_key = max(valid_keys,
                       key=lambda k: (metrics_grid[k]["ffds"],
                                      -metrics_grid[k]["independence"]))
    else:
        best_key = None

    # --- Adaptive x-axis range ---
    if fault_freqs:
        max_fault_freq = max(fault_freqs.values())
        x_max = min(max_fault_freq * 2.5, fs / 2)
    else:
        x_max = min(fs / 2, 800)

    # --- Build figure ---
    fig = plt.figure(figsize=(n_cols * 5.0, n_rows * 2.8 + 4.5))

    # Envelope spectra (top half)
    spec_y0 = 0.48
    spec_height = 0.52
    cell_h = spec_height / n_rows
    cell_w = 0.92 / n_cols

    for i, tfa in enumerate(tfa_methods):
        for j, bss in enumerate(bss_methods):
            key = (tfa, bss)
            left = 0.06 + j * cell_w
            bottom = spec_y0 + (n_rows - 1 - i) * cell_h
            ax = fig.add_axes([left, bottom, cell_w * 0.9, cell_h * 0.82])

            if key in cross_results:
                S_est = cross_results[key]
                sig = S_est[0]
                analytic = hilbert(sig)
                envelope = np.abs(analytic)
                N = len(envelope)
                env_spec = np.abs(np.fft.rfft(envelope))
                freq = np.fft.rfftfreq(N, 1.0 / fs)
                ax.plot(freq, env_spec, linewidth=0.4, color="#1976D2")
                ax.set_xlim([0, x_max])
                max_y = np.percentile(env_spec[1:], 98)
                ax.set_ylim([0, max_y * 1.1])

                if fault_freqs is not None:
                    for fname, ff in fault_freqs.items():
                        ax.axvline(x=ff, color="#D32F2F", linestyle="--", alpha=0.5,
                                   linewidth=0.5)
                        for h in [2, 3]:
                            ax.axvline(x=h * ff, color="#D32F2F", linestyle=":",
                                       alpha=0.2, linewidth=0.4)

                # Source independence value (top-right annotation)
                ind = metrics_grid[key]["independence"]
                ax.text(0.98, 0.92, f"ind={ind:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.8))

                # FFDS value
                fd = metrics_grid[key]["ffds"]
                ax.text(0.98, 0.75, f"FFDS={fd:.1f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6, color="#D32F2F" if fd > 3 else "#555555",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.8))

                # Gold border for best
                if key == best_key:
                    for spine in ax.spines.values():
                        spine.set_color("#FFD700")
                        spine.set_linewidth(3)
                    ax.text(0.5, 1.05, "BEST", transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color="#B8860B")
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            if j > 0:
                ax.set_yticklabels([])
            if i < n_rows - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=5)

            if i == 0:
                ax.set_title(bss.upper(), fontsize=9, fontweight="bold",
                             color="#333333")
            if j == 0:
                ax.set_ylabel(tfa.upper(), fontsize=8, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Hz", fontsize=6)

    # --- Heatmaps (bottom half) ---
    heatmap_y0 = 0.04
    heatmap_height = 0.38
    hmap_w = 0.41

    # Build data arrays
    indep_arr = np.zeros((n_rows, n_cols))
    ffds_arr = np.zeros((n_rows, n_cols))
    for i, tfa in enumerate(tfa_methods):
        for j, bss in enumerate(bss_methods):
            key = (tfa, bss)
            indep_arr[i, j] = metrics_grid[key]["independence"]
            ffds_arr[i, j] = metrics_grid[key]["ffds"]

    # --- Independence heatmap (left) ---
    ax_ind = fig.add_axes([0.06, heatmap_y0, hmap_w, heatmap_height])
    im_ind = ax_ind.imshow(indep_arr, cmap="RdYlGn_r", aspect="auto",
                            vmin=0, vmax=0.5)
    ax_ind.set_xticks(range(n_cols))
    ax_ind.set_yticks(range(n_rows))
    ax_ind.set_xticklabels([b.upper() for b in bss_methods], fontsize=8)
    ax_ind.set_yticklabels([t.upper() for t in tfa_methods], fontsize=8)
    ax_ind.set_title("Source Independence (lower = better)",
                     fontsize=10, fontweight="bold")
    ax_ind.set_xlabel("BSS Method", fontsize=8)

    for i in range(n_rows):
        for j in range(n_cols):
            val = indep_arr[i, j]
            color = "white" if val > 0.25 else "black"
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:.3f}"
            ax_ind.text(j, i, text, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
            if (tfa_methods[i], bss_methods[j]) == best_key:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      fill=False, edgecolor="#FFD700",
                                      linewidth=4, linestyle="-",
                                      joinstyle="round", capstyle="round")
                ax_ind.add_patch(rect)

    fig.colorbar(im_ind, ax=ax_ind, fraction=0.046, pad=0.02,
                 label="Mean |Off-Diag Correlation|")

    # --- FFDS heatmap (right) ---
    ax_ffds = fig.add_axes([0.53, heatmap_y0, hmap_w, heatmap_height])
    vmax_ffds = np.nanmax(ffds_arr) * 1.1 if np.nanmax(ffds_arr) > 0 else 1.0
    im_ffds = ax_ffds.imshow(ffds_arr, cmap="YlGn", aspect="auto",
                              vmin=0, vmax=vmax_ffds)
    ax_ffds.set_xticks(range(n_cols))
    ax_ffds.set_yticks(range(n_rows))
    ax_ffds.set_xticklabels([b.upper() for b in bss_methods], fontsize=8)
    ax_ffds.set_yticklabels([])
    ax_ffds.set_title("Fault Detection Score  FFDS (higher = better)",
                      fontsize=10, fontweight="bold")
    ax_ffds.set_xlabel("BSS Method", fontsize=8)

    for i in range(n_rows):
        for j in range(n_cols):
            val = ffds_arr[i, j]
            vmax = vmax_ffds if vmax_ffds > 0 else 1.0
            color = "white" if val < vmax * 0.5 else "black"
            if np.isnan(val) or val == 0:
                text = "N/A"
            else:
                text = f"{val:.1f}"
            ax_ffds.text(j, i, text, ha="center", va="center",
                         fontsize=9, fontweight="bold", color=color)
            if (tfa_methods[i], bss_methods[j]) == best_key:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      fill=False, edgecolor="#FFD700",
                                      linewidth=4, linestyle="-",
                                      joinstyle="round", capstyle="round")
                ax_ffds.add_patch(rect)

    fig.colorbar(im_ffds, ax=ax_ffds, fraction=0.046, pad=0.02,
                 label="FFDS (peak/noise ratio)")

    # --- Global title ---
    fig.suptitle(f"{title_prefix}Separation Quality Report — TFA × BSS Cross-Validation",
                 fontsize=14, fontweight="bold", y=0.98)

    return fig, metrics_grid
