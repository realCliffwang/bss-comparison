"""
Configuration management for BSS-Test framework.

Provides:
- Default configurations for all experiments
- YAML/JSON config file loading
- Command-line argument parsing
- Configuration validation
- Environment variable overrides

Usage:
    from bss_test.utils.config import get_config, load_config

    # Get default config
    config = get_config("cwru")

    # Load from file
    config = load_config("config.yaml")

    # Merge configs
    config = merge_configs(default_config, user_config)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    detrend: bool = True
    bandpass: Optional[Tuple[float, float]] = (100, 5000)
    normalize: str = "zscore"  # "zscore", "minmax", or None
    resample_fs: Optional[float] = None
    filter_order: int = 4


@dataclass
class CWTConfig:
    """CWT/TFA configuration."""
    wavelet: str = "cmor1.5-1.0"
    n_bands: int = 20
    freq_range: Optional[Tuple[float, float]] = (100, 5000)
    mode: str = "single_channel_expansion"  # or "multi_channel"
    tfa_method: str = "cwt"  # cwt, stft, wpt, vmd, emd, eemd, ceemdan
    bands_per_ch: Optional[int] = None


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
    cwt: CWTConfig = field(default_factory=CWTConfig)
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
        return load_config(path)


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
        cwt=CWTConfig(
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
        cwt=CWTConfig(
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
        cwt=CWTConfig(
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


def get_config(dataset: str = "cwru") -> ExperimentConfig:
    """
    Get default configuration for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name: "cwru", "phm2010", or "nasa".

    Returns
    -------
    ExperimentConfig
        Default configuration for the specified dataset.

    Raises
    ------
    ValueError
        If dataset is not recognized.
    """
    dataset = dataset.lower()
    if dataset not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DEFAULT_CONFIGS.keys())}"
        )
    return DEFAULT_CONFIGS[dataset]


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load configuration from a YAML or JSON file.

    Parameters
    ----------
    config_path : str
        Path to configuration file (.yaml, .yml, or .json).

    Returns
    -------
    ExperimentConfig
        Loaded configuration.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    ValueError
        If file format is not supported.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config format: {suffix}. "
            f"Use .yaml, .yml, or .json"
        )

    return dict_to_config(data)


def dict_to_config(data: Dict[str, Any]) -> ExperimentConfig:
    """
    Convert dictionary to ExperimentConfig.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    ExperimentConfig
        Configuration object.
    """
    # Start with default config
    config = ExperimentConfig()

    # Update top-level fields
    for key in ["name", "dataset", "data_dir", "fault_type", "load", "channels",
                 "output_dir", "log_level", "log_file"]:
        if key in data:
            setattr(config, key, data[key])

    # Update sub-configurations
    if "preprocess" in data and isinstance(data["preprocess"], dict):
        for k, v in data["preprocess"].items():
            if hasattr(config.preprocess, k):
                setattr(config.preprocess, k, v)

    if "cwt" in data and isinstance(data["cwt"], dict):
        for k, v in data["cwt"].items():
            if hasattr(config.cwt, k):
                setattr(config.cwt, k, v)

    if "bss" in data and isinstance(data["bss"], dict):
        for k, v in data["bss"].items():
            if hasattr(config.bss, k):
                setattr(config.bss, k, v)

    if "features" in data and isinstance(data["features"], dict):
        for k, v in data["features"].items():
            if hasattr(config.features, k):
                setattr(config.features, k, v)

    if "classifier" in data and isinstance(data["classifier"], dict):
        for k, v in data["classifier"].items():
            if hasattr(config.classifier, k):
                setattr(config.classifier, k, v)

    if "visualization" in data and isinstance(data["visualization"], dict):
        for k, v in data["visualization"].items():
            if hasattr(config.visualization, k):
                setattr(config.visualization, k, v)

    if "feature_freqs" in data and isinstance(data["feature_freqs"], dict):
        config.feature_freqs = data["feature_freqs"]

    return config


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Convert ExperimentConfig to dictionary.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration object.

    Returns
    -------
    dict
        Configuration as dictionary.
    """
    return asdict(config)


def save_config(config: ExperimentConfig, filepath: str) -> None:
    """
    Save configuration to file.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to save.
    filepath : str
        Output file path (.yaml, .yml, or .json).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = config_to_dict(config)

    suffix = filepath.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML output. pip install pyyaml")
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
    elif suffix == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def merge_configs(
    base: ExperimentConfig,
    override: Dict[str, Any]
) -> ExperimentConfig:
    """
    Merge override values into base configuration.

    Parameters
    ----------
    base : ExperimentConfig
        Base configuration.
    override : dict
        Values to override.

    Returns
    -------
    ExperimentConfig
        Merged configuration.
    """
    base_dict = config_to_dict(base)
    deep_update(base_dict, override)
    return dict_to_config(base_dict)


def deep_update(base: dict, override: dict) -> dict:
    """
    Recursively update base dictionary with override values.

    Parameters
    ----------
    base : dict
        Base dictionary.
    override : dict
        Override values.

    Returns
    -------
    dict
        Updated dictionary.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    args : list or None
        Arguments to parse. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="BSS-Test: Bearing Fault Diagnosis with BSS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["cwru", "phm2010", "nasa"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory"
    )
    parser.add_argument(
        "--fault-type",
        type=str,
        help="Fault type (CWRU: inner_race_007, ball_007, outer_race_6_007)"
    )
    parser.add_argument(
        "--bss-method",
        type=str,
        choices=["SOBI", "FastICA", "JADE", "PICARD", "NMF", "PCA"],
        help="BSS method"
    )
    parser.add_argument(
        "--tfa-method",
        type=str,
        choices=["cwt", "stft", "wpt", "vmd", "emd", "eemd", "ceemdan"],
        help="Time-frequency analysis method"
    )
    parser.add_argument(
        "--n-sources",
        type=int,
        help="Number of sources to extract"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot display"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save current configuration to file"
    )

    return parser.parse_args(args)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """
    Create configuration from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    ExperimentConfig
        Configuration based on arguments.
    """
    # Start with base config
    if args.config:
        config = load_config(args.config)
    elif args.dataset:
        config = get_config(args.dataset)
    else:
        config = get_config("cwru")  # Default

    # Apply overrides
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.fault_type:
        config.fault_type = args.fault_type
    if args.bss_method:
        config.bss.method = args.bss_method
    if args.tfa_method:
        config.cwt.tfa_method = args.tfa_method
    if args.n_sources:
        config.bss.n_sources = args.n_sources
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.log_level:
        config.log_level = args.log_level
    if args.no_plots:
        config.visualization.show_plots = False

    return config


# Environment variable overrides
def apply_env_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """
    Apply environment variable overrides to configuration.

    Environment variables:
    - BSSTEST_DATA_DIR: Override data directory
    - BSSTEST_OUTPUT_DIR: Override output directory
    - BSSTEST_LOG_LEVEL: Override log level
    - BSSTEST_BSS_METHOD: Override BSS method

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to update.

    Returns
    -------
    ExperimentConfig
        Updated configuration.
    """
    env_mappings = {
        "BSSTEST_DATA_DIR": ("data_dir", str),
        "BSSTEST_OUTPUT_DIR": ("output_dir", str),
        "BSSTEST_LOG_LEVEL": ("log_level", str),
        "BSSTEST_BSS_METHOD": ("bss.method", str),
    }

    for env_var, (attr_path, type_func) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Navigate nested attributes
            parts = attr_path.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], type_func(value))

    return config
