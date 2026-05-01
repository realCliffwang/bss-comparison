"""
Pytest configuration and shared fixtures.

Provides common test fixtures for the BSS-Test test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_signal_1d():
    """Generate a simple 1D test signal."""
    np.random.seed(42)
    fs = 1000
    t = np.arange(1000) / fs
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    signal += 0.1 * np.random.randn(len(t))
    return signal


@pytest.fixture
def sample_signals_multi():
    """Generate multi-channel test signals."""
    np.random.seed(42)
    n_channels = 3
    n_samples = 2000
    fs = 1000
    t = np.arange(n_samples) / fs

    signals = np.zeros((n_channels, n_samples))
    signals[0] = np.sin(2 * np.pi * 30 * t)
    signals[1] = np.sin(2 * np.pi * 80 * t + np.pi / 4)
    signals[2] = 0.3 * np.random.randn(n_samples)

    return signals, fs


@pytest.fixture
def sample_bss_mixture():
    """Generate a synthetic BSS mixture with known sources."""
    np.random.seed(42)
    n_sources = 3
    n_obs = 5
    n_samples = 2000
    fs = 1000
    t = np.arange(n_samples) / fs

    # Create source signals
    S = np.zeros((n_sources, n_samples))
    S[0] = np.sin(2 * np.pi * 30 * t)
    S[1] = np.sign(np.sin(2 * np.pi * 5 * t))  # Square wave
    S[2] = np.random.randn(n_samples)

    # Create mixing matrix
    A = np.random.randn(n_obs, n_sources)

    # Mix signals
    X = A @ S

    return S, X, A, fs


@pytest.fixture
def sample_features():
    """Generate sample feature vectors for classification testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 15

    X = np.random.randn(n_samples, n_features)
    y = np.array(["normal"] * 50 + ["fault"] * 50)

    return X, y


@pytest.fixture
def sample_multiclass_features():
    """Generate multi-class sample features."""
    np.random.seed(42)
    n_samples = 120
    n_features = 15

    X = np.random.randn(n_samples, n_features)
    y = np.array(
        ["normal"] * 40 +
        ["inner_race"] * 40 +
        ["ball"] * 40
    )

    return X, y


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def cwru_config():
    """Get CWRU test configuration."""
    from src.config import get_config
    return get_config("cwru")


@pytest.fixture
def preprocess_config():
    """Get preprocessing test configuration."""
    from src.config import PreprocessConfig
    return PreprocessConfig(
        detrend=True,
        bandpass=(100, 5000),
        normalize="zscore",
    )


@pytest.fixture
def cwt_config():
    """Get CWT test configuration."""
    from src.config import CWTConfig
    return CWTConfig(
        wavelet="cmor1.5-1.0",
        n_bands=10,
        freq_range=(100, 5000),
        mode="single_channel_expansion",
    )


@pytest.fixture
def bss_config():
    """Get BSS test configuration."""
    from src.config import BSSConfig
    return BSSConfig(
        method="SOBI",
        n_sources=3,
        n_lags=10,
    )
