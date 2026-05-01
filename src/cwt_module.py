"""
Continuous Wavelet Transform (CWT) module and observation matrix construction.

Uses PyWavelets (pywt) for CWT computation.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import stft as sp_signal_stft


def cwt_transform(signal, fs, wavelet="cmor1.5-1.0", n_bands=20,
                  freq_range=None, max_scale=None):
    """
    Compute 1D Continuous Wavelet Transform.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Wavelet type. Recommended:
        - "cmor1.5-1.0": Complex Morlet, good for time-frequency analysis
        - "morl": Real Morlet
        - "cgau8": Complex Gaussian
    freq_range : tuple (low, high) or None
        Frequency range in Hz. If None, uses (20, fs/2).
    n_bands : int
        Number of frequency bands (scales).
    max_scale : int or None
        Maximum scale. If None, auto-computed.

    Returns
    -------
    coef : ndarray (n_bands, n_samples)
        CWT coefficient magnitudes (absolute value).
    freqs : ndarray (n_bands,)
        Frequencies corresponding to each band.
    scales : ndarray (n_bands,)
        Scales used in CWT.
    """
    N = len(signal)

    if freq_range is None:
        f_low = 20.0
        f_high = fs / 2.0
    else:
        f_low, f_high = freq_range
        f_low = max(f_low, 0.1)
        f_high = min(f_high, fs / 2.0)

    # Convert frequency range to scale range
    # For Morlet-type wavelets: scale = center_freq * fs / (f * sampling_period)
    # pywt.cwt uses: frequency = center_frequency / (scale * sampling_period)
    # where sampling_period = 1/fs
    center_freq = pywt.scale2frequency(wavelet, 1)
    # frequency = center_freq / (scale * dt) = center_freq * fs / scale
    # => scale = center_freq * fs / frequency

    scale_low = center_freq * fs / f_high    # smaller frequency -> larger scale
    scale_high = center_freq * fs / f_low

    # Logarithmic spacing for scales (better frequency resolution at low freq)
    scales = np.logspace(np.log10(scale_low), np.log10(scale_high), n_bands)

    if max_scale is not None:
        scales = np.minimum(scales, max_scale)

    # Handle edge case where scales may be rounded to 1
    scales = np.maximum(scales, 1.0)
    scales = np.unique(scales.astype(int)).astype(float)

    # pywt.cwt returns: coef shape (n_scales, n_samples)
    # Note: pywt 1.4+ API: pywt.cwt(signal, scales, wavelet)
    # Older API: pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    try:
        coef, _freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs)
    except TypeError:
        # Fallback for older pywt versions
        coef, _freqs = pywt.cwt(signal, scales, wavelet)

    # Take magnitude (for complex wavelets; for real wavelets abs-value)
    coef_mag = np.abs(coef)

    # Get frequencies array
    if len(_freqs) == len(scales):
        freqs = _freqs
    else:
        freqs = center_freq * fs / scales

    return coef_mag, freqs, scales


def cwt_transform_multichannel(signals, fs, **kwargs):
    """
    Apply CWT to each channel of a multi-channel signal.

    Parameters
    ----------
    signals : ndarray (n_channels, n_samples)
        Multi-channel signal.
    fs : float
        Sampling rate.
    **kwargs : dict
        Passed to cwt_transform().

    Returns
    -------
    cwt_results : list of dict
        Each dict contains: coef, freqs, scales, channel_idx.
    """
    n_channels = signals.shape[0]
    results = []
    for ch in range(n_channels):
        coef, freqs, scales = cwt_transform(signals[ch], fs, **kwargs)
        results.append({
            "coef": coef,
            "freqs": freqs,
            "scales": scales,
            "channel_idx": ch,
        })
    return results


def build_observation_matrix(signals_pre, fs, config=None):
    """
    Build observation matrix X_for_bss via TFA + band selection.

    Two modes:
    1. "multi_channel": Each physical channel -> TFA -> select K bands ->
                        concatenate (n_ch * K, n_samples).
    2. "single_channel_expansion": Single channel -> TFA -> select K bands ->
                        each band as pseudo-channel -> (K, n_samples).

    Parameters
    ----------
    signals_pre : ndarray (n_channels, n_samples)
        Preprocessed signals.
    fs : float
        Sampling rate.
    config : dict
        Configuration with keys:
        - mode : "single_channel_expansion" or "multi_channel" (required)
        - tfa_method : "cwt" (default) | "stft" | "wpt" | "vmd" | "emd" |
                       "eemd" | "ceemdan"
        - n_bands : int (default 20)
        - freq_range : tuple (low, high) or None
        - wavelet : str (default "cmor1.5-1.0") — used by CWT and WPT
        - band_indices : list of int or None — explicit band selection
        - bands_per_ch : int — bands per channel in multi_channel mode
        - return_complex : bool — if True, use complex coefficients (CWT only)

    Returns
    -------
    X_for_bss : ndarray (n_obs, n_samples)
        Observation matrix for BSS.
    obs_labels : list of str
        Labels describing each observation (e.g., "ch0_f123Hz").
    """
    if config is None:
        config = {}

    mode = config.get("mode", "single_channel_expansion")
    n_bands = config.get("n_bands", 20)
    freq_range = config.get("freq_range", None)
    wavelet = config.get("wavelet", "cmor1.5-1.0")
    band_indices = config.get("band_indices", None)
    tfa_method = config.get("tfa_method", "cwt")

    if mode == "single_channel_expansion":
        signal_1d = signals_pre[0]  # Use first channel
        coef, freqs = time_freq_factory(
            signal_1d, fs, method=tfa_method,
            n_bands=n_bands, freq_range=freq_range, wavelet=wavelet
        )

        if band_indices is not None:
            coef = coef[band_indices]
        else:
            band_indices = list(range(coef.shape[0]))

        X_for_bss = coef  # shape (K, n_samples)
        obs_labels = [f"ch0_f{freqs[i]:.0f}Hz" for i in band_indices]

    elif mode == "multi_channel":
        n_channels = signals_pre.shape[0]
        bands_per_ch = config.get("bands_per_ch", n_bands // n_channels)
        blocks = []
        obs_labels = []

        for ch in range(n_channels):
            coef, freqs = time_freq_factory(
                signals_pre[ch], fs, method=tfa_method,
                n_bands=bands_per_ch, freq_range=freq_range, wavelet=wavelet
            )
            if band_indices is not None and ch == 0:
                ch_indices = band_indices
            else:
                ch_indices = list(range(coef.shape[0]))
            blocks.append(coef[ch_indices])
            obs_labels += [f"ch{ch}_f{freqs[i]:.0f}Hz" for i in ch_indices]

        X_for_bss = np.vstack(blocks)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single_channel_expansion' or 'multi_channel'.")

    return X_for_bss, obs_labels


def plot_cwt_spectrogram(coef, freqs, n_samples, fs, title=None, ax=None,
                         vmin=None, vmax=None, cmap="jet"):
    """
    Plot CWT spectrogram (time-frequency representation).

    Parameters
    ----------
    coef : ndarray (n_scales, n_samples)
        CWT coefficient matrix.
    freqs : ndarray (n_scales,)
        Frequencies in Hz.
    n_samples : int
        Number of time samples.
    fs : float
        Sampling rate in Hz.
    title : str or None
    ax : matplotlib axis or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    time_axis = np.arange(n_samples) / fs
    extent = [time_axis[0], time_axis[-1], freqs[0], freqs[-1]]

    # Normalize for display
    coef_db = 20 * np.log10(coef / (coef.max(axis=1, keepdims=True) + 1e-12) + 1e-12)

    im = ax.imshow(coef_db, aspect="auto", origin="lower", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude [dB]")
    return fig, ax


# ============================================================
# Additional Time-Frequency Analysis methods
# ============================================================


def stft_transform(signal, fs, nperseg=256, noverlap=192, nfft=None,
                   freq_range=None, n_bands=20, **kwargs):
    """
    Short-Time Fourier Transform via scipy.signal.stft.

    GitHub ref: https://github.com/scipy/scipy

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of overlap samples.
    nfft : int or None
        FFT length (defaults to nperseg).
    freq_range : tuple (low, high) or None
        Frequency range to extract (Hz). If None, uses full range.
    n_bands : int
        Number of frequency bands to return (selects top-energy bands if
        fewer than total STFT bins).

    Returns
    -------
    matrix : ndarray (n_bands, n_samples)
        STFT magnitude (absolute value), interpolated to match input length.
    freq_axes : ndarray (n_bands,)
        Frequency values for each output band.
    """
    if nfft is None:
        nfft = nperseg

    f, t_stft, Zxx = sp_signal_stft(signal, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap, nfft=nfft, **kwargs)
    Zxx_mag = np.abs(Zxx)  # shape (n_freq, n_time)

    # Select frequency range
    if freq_range is not None:
        f_low, f_high = freq_range
        mask = (f >= f_low) & (f <= f_high)
        Zxx_mag = Zxx_mag[mask]
        f = f[mask]

    n_freq_total = Zxx_mag.shape[0]

    # If more bins than n_bands, select top-energy bands
    if n_freq_total > n_bands:
        band_energy = np.mean(Zxx_mag**2, axis=1)
        top_idx = np.argsort(band_energy)[-n_bands:]
        top_idx = np.sort(top_idx)
        Zxx_mag = Zxx_mag[top_idx]
        f = f[top_idx]

    n_time_stft = Zxx_mag.shape[1]
    n_samples = len(signal)

    # Interpolate to match original signal length
    if n_time_stft != n_samples:
        t_orig = np.linspace(0, n_samples - 1, n_samples)
        t_stft_idx = np.linspace(0, n_samples - 1, n_time_stft)
        Zxx_interp = np.zeros((Zxx_mag.shape[0], n_samples))
        for i in range(Zxx_mag.shape[0]):
            Zxx_interp[i] = np.interp(t_orig, t_stft_idx, Zxx_mag[i])
        matrix = Zxx_interp
    else:
        matrix = Zxx_mag

    return matrix, f


def wpt_transform(signal, fs, wavelet="db4", max_level=4, n_bands=16,
                  **kwargs):
    """
    Wavelet Packet Transform via PyWavelets.

    GitHub ref: https://github.com/PyWavelets/pywt

    Extracts leaf-node reconstructed signals as pseudo-channels.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Wavelet name (e.g. "db4", "sym5").
    max_level : int
        Decomposition level.
    n_bands : int
        Max number of leaf nodes to return (top-energy nodes first).

    Returns
    -------
    matrix : ndarray (n_bands, n_samples)
        Reconstructed signals from each WPT leaf node.
    freq_axes : ndarray (n_bands,)
        Nominal center frequencies of each node band.
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="symmetric",
                            maxlevel=max_level)
    nodes = wp.get_level(max_level, "natural")
    leaf_names = [node.path for node in nodes]
    leaf_signals = np.array([wp[node_name].data for node_name in leaf_names])  # (n_leaves, n_len)

    # Each leaf covers a frequency sub-band: fs/2 split per level -> fs/2*(node_idx/(2**level))
    n_leaves = len(nodes)
    nyq = fs / 2.0
    bin_width = nyq / (2 ** max_level)

    # Select top-energy nodes
    node_energy = np.mean(leaf_signals**2, axis=1)
    top_k = min(n_bands, n_leaves)
    top_idx = np.argsort(node_energy)[-top_k:]
    top_idx = np.sort(top_idx)

    matrix = leaf_signals[top_idx]

    # Calculate frequency axes from node paths
    # PyWavelets uses 'a' and 'd' for approximation and detail coefficients
    freq_axes = []
    for i in top_idx:
        path = leaf_names[i]
        # Convert path to binary index: 'a' -> 0, 'd' -> 1
        binary_str = path.replace('a', '0').replace('d', '1')
        node_idx = int(binary_str, 2)
        freq_axes.append((node_idx + 0.5) * bin_width)
    freq_axes = np.array(freq_axes)

    return matrix, freq_axes


def vmd_transform(signal, fs, K=6, alpha=2000, tau=0, DC=0, init=1, tol=1e-7,
                  **kwargs):
    """
    Variational Mode Decomposition.

    GitHub ref: https://github.com/vrcarva/vmdpy
    Fallback: sktime.libs.vmdpy

    Dragomiretskiy & Zosso (2014) "Variational Mode Decomposition,"
    IEEE Trans. Signal Processing.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    K : int
        Number of modes to decompose into (typical: 4, 6, 8).
    alpha : float
        Bandwidth constraint parameter.
    tau : float
        Noise tolerance (0 for no noise mode).
    DC : int
        Include DC mode (0 or 1).
    init : int
        Initialization method (0=all zeros, 1=uniform distributed, 2=random).
    tol : float
        Convergence tolerance.

    Returns
    -------
    matrix : ndarray (K, n_samples)
        Decomposed modes as pseudo-channels.
    freq_axes : ndarray (K,)
        Center frequencies of each mode (Hz).
    """
    _vmd = None
    try:
        from vmdpy import VMD
        _vmd = VMD
    except ImportError:
        try:
            from sktime.libs.vmdpy import VMD
            _vmd = VMD
        except ImportError:
            raise ImportError(
                "VMD requires vmdpy or sktime. Install with:\n"
                "  pip install vmdpy\n"
                "or\n"
                "  pip install sktime"
            )

    N = len(signal)
    # vmdpy.VMD expects (signal, alpha, tau, K, DC, init, tol)
    u, u_hat, omega = _vmd(signal, alpha=alpha, tau=tau, K=K, DC=DC,
                            init=init, tol=tol)

    # u shape: (K, N)
    matrix = u.astype(np.float64)

    # omega: center frequencies in normalized rad/sample
    # Convert to Hz: omega * fs / (2*pi)
    center_freqs = omega * fs / (2 * np.pi)

    return matrix, center_freqs


def emd_transform(signal, fs, max_imf=6, **kwargs):
    """
    Empirical Mode Decomposition via PyEMD.

    GitHub ref: https://github.com/laszukdawid/PyEMD
    Install: pip install EMD-signal

    Extract first N IMF components as pseudo-channels.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    max_imf : int
        Maximum number of IMFs to extract.

    Returns
    -------
    matrix : ndarray (max_imf, n_samples)
        First N IMF components.
    freq_axes : ndarray (max_imf,)
        Mean instantaneous frequency of each IMF (Hz), via Hilbert transform.
    """
    try:
        from PyEMD import EMD
    except ImportError:
        raise ImportError(
            "EMD requires PyEMD. Install with: pip install EMD-signal\n"
            "GitHub: https://github.com/laszukdawid/PyEMD"
        )

    emd = EMD()
    imfs = emd.emd(signal, max_imf=max_imf)

    n_imfs = min(imfs.shape[0], max_imf)
    if imfs.ndim == 2:
        imfs = imfs[:n_imfs]
    else:
        imfs = imfs.reshape(1, -1)[:n_imfs]

    # Estimate mean frequency of each IMF via Hilbert transform
    from scipy.signal import hilbert as sp_hilbert
    mean_freqs = np.zeros(n_imfs)
    for i in range(n_imfs):
        analytic = sp_hilbert(imfs[i])
        phase = np.unwrap(np.angle(analytic))
        if len(phase) > 1:
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            mean_freqs[i] = float(np.mean(np.abs(inst_freq)))
        else:
            mean_freqs[i] = 0.0

    matrix = np.array(imfs, dtype=np.float64)
    return matrix, mean_freqs


def eemd_transform(signal, fs, max_imf=6, trials=100, noise_std=0.2, **kwargs):
    """
    Ensemble Empirical Mode Decomposition via PyEMD.

    GitHub ref: https://github.com/laszukdawid/PyEMD
    Install: pip install EMD-signal

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate.
    max_imf : int
        Max IMFs to extract.
    trials : int
        Number of ensemble trials.
    noise_std : float
        Standard deviation of added noise relative to signal std.

    Returns
    -------
    matrix : ndarray (max_imf, n_samples)
    freq_axes : ndarray (max_imf,)
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        raise ImportError(
            "EEMD requires PyEMD. Install with: pip install EMD-signal\n"
            "GitHub: https://github.com/laszukdawid/PyEMD"
        )

    eemd = EEMD(trials=trials, noise_width=noise_std)
    imfs = eemd.eemd(signal, max_imf=max_imf)

    n_imfs = min(imfs.shape[0], max_imf)
    if imfs.ndim == 2:
        imfs = imfs[:n_imfs]
    else:
        imfs = imfs.reshape(1, -1)[:n_imfs]

    from scipy.signal import hilbert as sp_hilbert
    mean_freqs = np.zeros(n_imfs)
    for i in range(n_imfs):
        analytic = sp_hilbert(imfs[i])
        phase = np.unwrap(np.angle(analytic))
        if len(phase) > 1:
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            mean_freqs[i] = float(np.mean(np.abs(inst_freq)))
        else:
            mean_freqs[i] = 0.0

    matrix = np.array(imfs, dtype=np.float64)
    return matrix, mean_freqs


def ceemdan_transform(signal, fs, max_imf=6, **kwargs):
    """
    Complete Ensemble EMD with Adaptive Noise via PyEMD.

    GitHub ref: https://github.com/laszukdawid/PyEMD
    Install: pip install EMD-signal

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate.
    max_imf : int
        Max IMFs to extract.

    Returns
    -------
    matrix : ndarray (max_imf, n_samples)
    freq_axes : ndarray (max_imf,)
    """
    try:
        from PyEMD import CEEMDAN
    except ImportError:
        raise ImportError(
            "CEEMDAN requires PyEMD. Install with: pip install EMD-signal\n"
            "GitHub: https://github.com/laszukdawid/PyEMD"
        )

    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(signal, max_imf=max_imf)

    n_imfs = min(imfs.shape[0], max_imf)
    if imfs.ndim == 2:
        imfs = imfs[:n_imfs]
    else:
        imfs = imfs.reshape(1, -1)[:n_imfs]

    from scipy.signal import hilbert as sp_hilbert
    mean_freqs = np.zeros(n_imfs)
    for i in range(n_imfs):
        analytic = sp_hilbert(imfs[i])
        phase = np.unwrap(np.angle(analytic))
        if len(phase) > 1:
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            mean_freqs[i] = float(np.mean(np.abs(inst_freq)))
        else:
            mean_freqs[i] = 0.0

    matrix = np.array(imfs, dtype=np.float64)
    return matrix, mean_freqs


def time_freq_factory(signal, fs, method="cwt", **kwargs):
    """
    Unified time-frequency analysis factory.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        1D time-domain signal.
    fs : float
        Sampling rate in Hz.
    method : str
        One of:
          "cwt"      - Continuous Wavelet Transform (PyWavelets)
          "stft"     - Short-Time Fourier Transform (scipy)
          "wpt"      - Wavelet Packet Transform (PyWavelets)
          "vmd"      - Variational Mode Decomposition (vmdpy)
          "emd"      - Empirical Mode Decomposition (PyEMD)
          "eemd"     - Ensemble EMD (PyEMD)
          "ceemdan"  - Complete Ensemble EMD with Adaptive Noise (PyEMD)
    **kwargs :
        Passed to the specific TFA function.

    Returns
    -------
    matrix : ndarray (n_bands, n_samples)
        Time-frequency representation as pseudo-channel matrix.
    freq_axes : ndarray (n_bands,)
        Frequency values for each band.
    """
    method = method.lower()

    if method == "cwt":
        n_bands = kwargs.pop("n_bands", 20)
        freq_range = kwargs.pop("freq_range", None)
        wavelet = kwargs.pop("wavelet", "cmor1.5-1.0")
        coef, freqs, _scales = cwt_transform(
            signal, fs, wavelet=wavelet, n_bands=n_bands, freq_range=freq_range
        )
        return coef, freqs

    elif method == "stft":
        kwargs.pop("wavelet", None)
        kwargs.pop("bands_per_ch", None)
        return stft_transform(signal, fs, **kwargs)

    elif method == "wpt":
        kwargs.pop("bands_per_ch", None)
        return wpt_transform(signal, fs, **kwargs)

    elif method == "vmd":
        kwargs.pop("wavelet", None)
        kwargs.pop("freq_range", None)
        kwargs.pop("bands_per_ch", None)
        return vmd_transform(signal, fs, **kwargs)

    elif method == "emd":
        kwargs.pop("wavelet", None)
        kwargs.pop("freq_range", None)
        kwargs.pop("bands_per_ch", None)
        return emd_transform(signal, fs, **kwargs)

    elif method == "eemd":
        kwargs.pop("wavelet", None)
        kwargs.pop("freq_range", None)
        kwargs.pop("bands_per_ch", None)
        return eemd_transform(signal, fs, **kwargs)

    elif method == "ceemdan":
        kwargs.pop("wavelet", None)
        kwargs.pop("freq_range", None)
        kwargs.pop("bands_per_ch", None)
        return ceemdan_transform(signal, fs, **kwargs)

    else:
        raise ValueError(
            f"Unknown TFA method: {method}. "
            f"Supported: cwt, stft, wpt, vmd, emd, eemd, ceemdan"
        )
