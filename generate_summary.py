"""
Generate PHM 2010 summary comparison figures across all 6 cutters.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import load_phm_cut, load_phm_wear
from preprocessing import preprocess_signals
from cwt_module import build_observation_matrix
from bss_module import run_bss
from evaluation import plot_correlation_matrix

CONFIG = {
    "data_dir": "data/phm2010_milling",
    "sensor_types": ["vib_x", "vib_y", "vib_z"],
    "preprocess": {"detrend": True, "bandpass": (100, 20000), "normalize": "zscore"},
    "cwt": {"wavelet": "cmor1.5-1.0", "n_bands": 20, "freq_range": (100, 20000),
            "bands_per_ch": 7, "mode": "multi_channel"},
    "bss": {"method": "SOBI", "n_sources": 5, "n_lags": 50},
    "output": "outputs/phm2010/_summary",
}
os.makedirs(CONFIG["output"], exist_ok=True)

BSS_CACHE = {}

def get_bss(tool_id):
    key = (tool_id,)
    if key in BSS_CACHE:
        return BSS_CACHE[key]
    signals, fs, _ = load_phm_cut(tool_id=tool_id, cut_no=150,
        data_dir=CONFIG["data_dir"], sensor_types=CONFIG["sensor_types"])
    n_use = min(signals.shape[1], int(1.0 * fs))
    signals = signals[:, :n_use]
    signals_pre, fs_pre = preprocess_signals(signals, fs, CONFIG["preprocess"])
    X_for_bss, _ = build_observation_matrix(signals_pre, fs_pre, CONFIG["cwt"])
    S_est, W = run_bss(X_for_bss, **CONFIG["bss"])
    BSS_CACHE[key] = (S_est, fs_pre, signals_pre)
    return S_est, fs_pre, signals_pre


# ================================================================
# Figure 1: Wear evolution — 3 training cutters side by side
# ================================================================
print("[1/4] Wear evolution comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, tool_id in enumerate([1, 4, 6]):
    ax = axes[idx]
    wear = load_phm_wear(tool_id, CONFIG["data_dir"])
    wear = wear[~np.isnan(wear)]

    # Sample ~15 cuts evenly
    sample_cuts = np.linspace(1, len(wear), 15, dtype=int)
    energies = np.zeros((CONFIG["bss"]["n_sources"], len(sample_cuts)))

    for j, cut_no in enumerate(sample_cuts):
        try:
            sig_cut, fs_cut, _ = load_phm_cut(
                tool_id=tool_id, cut_no=cut_no,
                data_dir=CONFIG["data_dir"], sensor_types=CONFIG["sensor_types"])
            n_u = min(sig_cut.shape[1], int(0.5 * fs_cut))
            sig_cut = sig_cut[:, :n_u]
            sig_p, fs_p = preprocess_signals(sig_cut, fs_cut, CONFIG["preprocess"])
            X_obs, _ = build_observation_matrix(sig_p, fs_p, CONFIG["cwt"])
            S_run, _ = run_bss(X_obs, **CONFIG["bss"])
            for s in range(CONFIG["bss"]["n_sources"]):
                energies[s, j] = np.sqrt(np.mean(S_run[s] ** 2))
        except Exception as e:
            energies[:, j] = np.nan

    wear_vals = wear[sample_cuts - 1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, CONFIG["bss"]["n_sources"]))
    for s in range(CONFIG["bss"]["n_sources"]):
        ax.plot(wear_vals, energies[s], "o-", markersize=4, linewidth=0.8,
                color=colors[s], label=f"S{s}")

    ax.set_xlabel("Wear VB [10^-3 mm]")
    ax.set_ylabel("Source RMS Energy")
    ax.set_title(f"Cutter c{tool_id}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle("PHM 2010 — Source Energy vs Tool Wear (Training Cutters)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(CONFIG["output"], "summary_wear_evolution.png"), dpi=150)
plt.close(fig)
print("  Done: summary_wear_evolution.png")


# ================================================================
# Figure 2: Source correlation matrix grid for all 6 cutters
# ================================================================
print("[2/4] Source correlation matrix grid...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, tool_id in enumerate(range(1, 7)):
    ax = axes[idx // 3, idx % 3]
    S_est, _, _ = get_bss(tool_id)
    corr = np.corrcoef(S_est)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(S_est.shape[0]))
    ax.set_yticks(range(S_est.shape[0]))
    label = "train" if tool_id in [1, 4, 6] else "test"
    ax.set_title(f"c{tool_id} ({label})")
    if idx >= 3:
        ax.set_xlabel("Source index")
    if idx % 3 == 0:
        ax.set_ylabel("Source index")
    for i in range(S_est.shape[0]):
        for j in range(S_est.shape[0]):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr[i,j]) > 0.5 else "black")

fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
fig.suptitle("PHM 2010 — Source Correlation Matrices (All 6 Cutters)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(CONFIG["output"], "summary_correlation_grid.png"), dpi=150)
plt.close(fig)
print("  Done: summary_correlation_grid.png")


# ================================================================
# Figure 3: Wear amplitude (VB) curves for training cutters
# ================================================================
print("[3/4] Wear curves comparison...")
fig, ax = plt.subplots(figsize=(10, 5))
colors = {1: "#2196F3", 4: "#FF9800", 6: "#4CAF50"}
for tool_id in [1, 4, 6]:
    wear = load_phm_wear(tool_id, CONFIG["data_dir"])
    wear = wear[~np.isnan(wear)]
    ax.plot(np.arange(1, len(wear)+1), wear, linewidth=1.5,
            color=colors[tool_id], label=f"c{tool_id}")

ax.set_xlabel("Cut Number")
ax.set_ylabel("Flank Wear VB [10^-3 mm]")
ax.set_title("PHM 2010 — Tool Wear Progression (Training Cutters)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG["output"], "summary_wear_curves.png"), dpi=150)
plt.close(fig)
print("  Done: summary_wear_curves.png")


# ================================================================
# Figure 4: Source energy distribution comparison across cutters
# ================================================================
print("[4/4] Source energy distribution...")
fig, ax = plt.subplots(figsize=(12, 5))
n_src = CONFIG["bss"]["n_sources"]
width = 0.12
x = np.arange(n_src)

for idx, tool_id in enumerate(range(1, 7)):
    S_est, _, _ = get_bss(tool_id)
    rms = np.sqrt(np.mean(S_est**2, axis=1))
    rms_norm = rms / rms.sum()
    offset = (idx - 2.5) * width
    label = f"c{tool_id}"
    ax.bar(x + offset, rms_norm, width, label=label, alpha=0.85)

ax.set_xlabel("Source Index")
ax.set_ylabel("Normalized RMS Energy")
ax.set_title("PHM 2010 — Source Energy Distribution (All 6 Cutters)")
ax.set_xticks(x)
ax.legend(fontsize=8, ncol=6)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(CONFIG["output"], "summary_energy_distribution.png"), dpi=150)
plt.close(fig)
print("  Done: summary_energy_distribution.png")

print(f"\nAll summary plots saved to {CONFIG['output']}/")
