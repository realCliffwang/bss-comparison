"""
BSS-Test: Bearing Fault Diagnosis with Blind Source Separation

A comprehensive framework for vibration-based fault diagnosis using
Blind Source Separation (BSS) techniques combined with Time-Frequency Analysis.

Modules:
    - data_loader: Data loading utilities for CWRU, PHM 2010, NASA datasets
    - preprocessing: Signal preprocessing (detrend, filter, normalize)
    - cwt_module: Time-Frequency Analysis (CWT, STFT, WPT, VMD, EMD, etc.)
    - bss_module: Blind Source Separation (SOBI, FastICA, JADE, PICARD, NMF, PCA)
    - feature_extractor: Feature extraction (time, frequency, time-frequency domain)
    - ml_classifier: Machine learning classifiers (SVM, RF, XGBoost, KNN, LDA)
    - dl_classifier: Deep learning classifiers (CNN, LSTM, Transformer)
    - evaluation: Evaluation metrics and visualization
    - config: Configuration management
    - logger: Logging utilities
    - exceptions: Custom exception classes
    - utils: Utility functions

Usage:
    from src.data_loader import load_cwru
    from src.preprocessing import preprocess_signals
    from src.cwt_module import build_observation_matrix
    from src.bss_module import run_bss
    from src.evaluation import evaluate_bss
"""

__version__ = "0.2.0"
__author__ = "BSS-Test Contributors"

# Import main components for easy access
from src.config import (
    ExperimentConfig,
    get_config,
    load_config,
    save_config,
)

from src.logger import (
    get_logger,
    setup_logging,
)

from src.exceptions import (
    BSSTestError,
    DataLoadError,
    PreprocessingError,
    TFAError,
    BSSError,
    FeatureExtractionError,
    ClassifierError,
    ConfigurationError,
)

# Version info
VERSION_INFO = {
    "version": __version__,
    "major": 0,
    "minor": 2,
    "patch": 0,
}


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> dict:
    """Get detailed version information."""
    return VERSION_INFO.copy()
