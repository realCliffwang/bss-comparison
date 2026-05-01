"""
Utility functions: synthetic data generation for testing when real data unavailable.

Use generate_synthetic_mixture() for testing the BSS pipeline without real datasets.
"""

import numpy as np


def generate_synthetic_mixture(n_sources=3, n_obs=5, n_samples=20000, fs=1000,
                                source_type="vibration", seed=42):
    """
    Generate synthetic BSS mixture for pipeline testing.

    Creates realistic vibration-like source signals and mixes them to produce
    observations, with known ground truth.

    Parameters
    ----------
    n_sources : int
        Number of true source signals.
    n_obs : int
        Number of observation channels.
    n_samples : int
        Number of time samples.
    fs : float
        Sampling rate in Hz.
    source_type : str
        "vibration": bearing/gear-like impulsive + harmonic signals
        "milling": CNC milling-like with spindle harmonics + noise
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    S_true : ndarray (n_sources, n_samples)
        Ground truth source signals.
    X_mixed : ndarray (n_obs, n_samples)
        Mixed observation signals.
    A : ndarray (n_obs, n_sources)
        Mixing matrix.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    S_true = np.zeros((n_sources, n_samples))

    if source_type == "vibration":
        # Source 0: Harmonic (simulating shaft rotation)
        f0 = 30.0  # Hz
        S_true[0] = (np.sin(2 * np.pi * f0 * t) +
                     0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                     0.3 * np.sin(2 * np.pi * 3 * f0 * t))

        # Source 1: Impulsive (simulating bearing fault impacts)
        impulse_period = int(fs / 100)  # 100 Hz impulse rate
        for i in range(0, n_samples, impulse_period):
            decay_len = min(50, n_samples - i)
            S_true[1, i:i + decay_len] = np.exp(-np.arange(decay_len) / 10)
        S_true[1] += 0.02 * rng.randn(n_samples)

        # Source 2: Gaussian noise (background vibration)
        S_true[2] = 0.5 * rng.randn(n_samples)

        if n_sources > 3:
            for s in range(3, n_sources):
                freq = 50 + s * 30
                S_true[s] = np.sin(2 * np.pi * freq * t) + 0.1 * rng.randn(n_samples)

    elif source_type == "milling":
        # Simulating CNC milling multi-channel sources
        spindle_freq = 173.0  # Hz (10400 RPM)

        # Source 0: Spindle fundamental + harmonics
        S_true[0] = (np.sin(2 * np.pi * spindle_freq * t) +
                     0.3 * np.sin(2 * np.pi * 2 * spindle_freq * t) +
                     0.15 * np.sin(2 * np.pi * 3 * spindle_freq * t))

        # Source 1: Tooth passing frequency (e.g., 2-flute cutter: 2×spindle)
        tpf = spindle_freq * 2  # tooth passing frequency
        S_true[1] = (np.sin(2 * np.pi * tpf * t) +
                     0.4 * np.sin(2 * np.pi * 2 * tpf * t))
        # Add amplitude modulation simulating chip thickness variation
        S_true[1] *= (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))

        # Source 2: Tool wear signal (low frequency trend + random)
        S_true[2] = 0.3 * t / t[-1]  # Increasing trend
        S_true[2] += 0.1 * np.sin(2 * np.pi * spindle_freq * 0.1 * t)  # Sub-harmonic
        S_true[2] += 0.05 * rng.randn(n_samples)

        # Source 3: Random vibration noise
        S_true[3] = rng.randn(n_samples)

        if n_sources > 4:
            for s in range(4, n_sources):
                freq = 200 + s * 100
                S_true[s] = np.sin(2 * np.pi * freq * t) + 0.05 * rng.randn(n_samples)

    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    # Generate random mixing matrix
    A = rng.randn(n_obs, n_sources)

    # Mix sources
    X_mixed = A @ S_true

    return S_true, X_mixed, A


def generate_phm_like_cut(tool_id=1, cut_no=1, n_samples=50000, fs=50000, seed=None):
    """
    Generate synthetic PHM-2010-like single-cut data.

    Simulates 7-channel data mimicking:
      force_x, force_y, force_z, vib_x, vib_y, vib_z, AE_rms

    Parameters
    ----------
    tool_id : int
    cut_no : int
    n_samples : int
    fs : int
    seed : int or None

    Returns
    -------
    signals : ndarray (7, n_samples) — 7 simulated channels
    fs : int
    wear : float — simulated wear for this cut
    """
    if seed is None:
        seed = tool_id * 1000 + cut_no
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs

    spindle_freq = 173.0  # Hz
    tpf = spindle_freq * 2  # tooth passing freq (2-flute)

    # Simulated wear: increases with cut number
    wear = 20 + 180 * (cut_no / 315) ** 2  # wear in 10^-3 mm, ~20 to ~200

    signals = np.zeros((7, n_samples))

    # Force channels (X, Y, Z) — dominated by tooth passing freq
    for i in range(3):
        base = np.sin(2 * np.pi * tpf * t + i * np.pi / 3)
        # Add wear effect: amplitude modulation increases with cut_no
        amp = 1.0 + 0.5 * (cut_no / 315)
        signals[i] = amp * base + 0.1 * rng.randn(n_samples)

    # Vibration channels (X, Y, Z) — spindle freq + harmonics + impulsive
    for i in range(3):
        base = (np.sin(2 * np.pi * spindle_freq * t + i * 1.5) +
                0.4 * np.sin(2 * np.pi * tpf * t + i * 0.8) +
                0.2 * np.sin(2 * np.pi * 3 * spindle_freq * t))
        # Add wear-related high-freq energy
        noise_level = 0.1 + 0.3 * (cut_no / 315)
        signals[3 + i] = base + noise_level * rng.randn(n_samples)

    # AE-RMS (acoustic emission) — sensitive to wear
    ae_base = 0.3 * (1 + cut_no / 315) * np.abs(rng.randn(n_samples))
    # Lowpass filter approximation via moving average
    window = 50
    for i in range(window, n_samples):
        signals[6, i] = np.mean(ae_base[i - window:i])

    return signals, fs, wear
