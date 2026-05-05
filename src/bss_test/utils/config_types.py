"""
Configuration dataclass definitions for BSS-Test framework.

Defines all configuration dataclasses used across the framework.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    detrend: bool = True
    bandpass: Optional[Tuple[float, float]] = (100, 5000)
    normalize: str = "zscore"  # "zscore", "minmax", or None
    resample_fs: Optional[float] = None
    filter_order: int = 4


@dataclass
class TFAConfig:
    """TFA (Time-Frequency Analysis) configuration."""
    wavelet: str = "cmor1.5-1.0"
    n_bands: int = 20
    freq_range: Optional[Tuple[float, float]] = (100, 5000)
    mode: str = "single_channel_expansion"  # or "multi_channel"
    tfa_method: str = "cwt"  # cwt, stft, wpt, vmd, emd, eemd, ceemdan
    bands_per_ch: Optional[int] = None


# Backward compatibility alias
CWTConfig = TFAConfig


@dataclass
class BSSConfig:
    """BSS configuration."""
    method: str = "SOBI"  # SOBI, FastICA, JADE, PICARD, NMF, PCA
    n_sources: int = 5
    n_lags: int = 50  # SOBI only
    max_iter: int = 2000
    tol: float = 1e-6
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    feature_set: str = "all"  # all, time, freq, time_freq
    wavelet: str = "db4"
    max_level: int = 4
    nperseg: int = 256
    noverlap: int = 192
    n_freq_bands: int = 32


@dataclass
class ClassifierConfig:
    """ML Classifier configuration."""
    method: str = "svm"  # svm, rf, xgb, knn, lda
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    cache_models: bool = True


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    style: str = "default"
    save_format: str = "png"
    show_plots: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Dataset
    name: str = "experiment"
    dataset: str = "cwru"  # cwru, phm2010, nasa
    data_dir: str = "data/cwru"
    fault_type: str = "inner_race_007"
    load: int = 0
    channels: List[str] = field(default_factory=lambda: ["DE"])

    # Sub-configurations
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tfa: TFAConfig = field(default_factory=TFAConfig)
    bss: BSSConfig = field(default_factory=BSSConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Output
    output_dir: str = "outputs"
    log_level: str = "info"
    log_file: Optional[str] = None

    # Bearing characteristic frequencies (CWRU example)
    feature_freqs: Dict[str, float] = field(default_factory=lambda: {
        "BPFO": 107.3,
        "BPFI": 162.2,
        "BSF": 70.6,
    })

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从 YAML 文件加载配置"""
        from bss_test.utils.config_io import load_config
        return load_config(path)

    @property
    def cwt(self) -> TFAConfig:
        """Backward compatibility: config.cwt -> config.tfa"""
        return self.tfa

    @cwt.setter
    def cwt(self, value: TFAConfig):
        self.tfa = value


# Default configurations for different datasets
DEFAULT_CONFIGS = {
    "cwru": ExperimentConfig(
        dataset="cwru",
        data_dir="data/cwru",
        fault_type="inner_race_007",
        load=0,
        channels=["DE"],
        preprocess=PreprocessConfig(
            detrend=True,
            bandpass=(100, 5000),
            normalize="zscore",
        ),
        tfa=TFAConfig(
            wavelet="cmor1.5-1.0",
            n_bands=20,
            freq_range=(100, 5000),
            mode="single_channel_expansion",
        ),
        bss=BSSConfig(
            method="SOBI",
            n_sources=5,
            n_lags=50,
        ),
        output_dir="outputs/cwru",
        feature_freqs={
            "BPFO": 107.3,
            "BPFI": 162.2,
            "BSF": 70.6,
        },
    ),
    "phm2010": ExperimentConfig(
        dataset="phm2010",
        data_dir="data/phm2010_milling",
        channels=["vib_x", "vib_y", "vib_z"],
        preprocess=PreprocessConfig(
            detrend=True,
            bandpass=(100, 20000),
            normalize="zscore",
        ),
        tfa=TFAConfig(
            wavelet="cmor1.5-1.0",
            n_bands=20,
            freq_range=(100, 20000),
            mode="multi_channel",
            bands_per_ch=7,
        ),
        bss=BSSConfig(
            method="SOBI",
            n_sources=6,
            n_lags=50,
        ),
        output_dir="outputs/phm2010",
    ),
    "nasa": ExperimentConfig(
        dataset="nasa",
        data_dir="data/phm2010_milling",
        channels=["vib_table", "vib_spindle", "force_ac"],
        preprocess=PreprocessConfig(
            detrend=True,
            bandpass=(10, 100),
            normalize="zscore",
        ),
        tfa=TFAConfig(
            wavelet="cmor1.5-1.0",
            n_bands=20,
            freq_range=(10, 120),
            mode="multi_channel",
            bands_per_ch=7,
        ),
        bss=BSSConfig(
            method="SOBI",
            n_sources=6,
            n_lags=50,
        ),
        output_dir="outputs/nasa_milling",
    ),
}
