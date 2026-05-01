"""
Custom exceptions for BSS-Test framework.

Provides specific exception types for better error handling and debugging.

Usage:
    from src.exceptions import BSSError, DataLoadError, PreprocessingError
    raise DataLoadError("File not found", filepath="data/cwru/122.mat")
"""


class BSSTestError(Exception):
    """Base exception for all BSS-Test errors."""

    def __init__(self, message: str, **kwargs):
        self.details = kwargs
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base} ({details_str})"
        return base


class DataLoadError(BSSTestError):
    """Error loading data from files."""
    pass


class DataNotFoundError(DataLoadError):
    """Data file not found."""
    pass


class DataFormatError(DataLoadError):
    """Data file format is invalid."""
    pass


class PreprocessingError(BSSTestError):
    """Error during signal preprocessing."""
    pass


class FilterError(PreprocessingError):
    """Error applying filter."""
    pass


class NormalizationError(PreprocessingError):
    """Error during normalization."""
    pass


class TFAError(BSSTestError):
    """Error during time-frequency analysis."""
    pass


class CWTError(TFAError):
    """Error in CWT computation."""
    pass


class STFTError(TFAError):
    """Error in STFT computation."""
    pass


class WPDError(TFAError):
    """Error in Wavelet Packet Decomposition."""
    pass


class VMDError(TFAError):
    """Error in Variational Mode Decomposition."""
    pass


class EMDError(TFAError):
    """Error in Empirical Mode Decomposition."""
    pass


class BSSError(BSSTestError):
    """Error during blind source separation."""
    pass


class ConvergenceError(BSSError):
    """BSS algorithm did not converge."""
    pass


class InvalidSourceCountError(BSSError):
    """Invalid number of sources requested."""
    pass


class FeatureExtractionError(BSSTestError):
    """Error during feature extraction."""
    pass


class ClassifierError(BSSTestError):
    """Error in ML classifier."""
    pass


class ModelNotTrainedError(ClassifierError):
    """Model has not been trained yet."""
    pass


class ConfigurationError(BSSTestError):
    """Error in configuration."""
    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value."""
    pass


class MissingConfigError(ConfigurationError):
    """Required configuration missing."""
    pass


class EvaluationError(BSSTestError):
    """Error during evaluation."""
    pass


class VisualizationError(BSSTestError):
    """Error generating visualization."""
    pass


# Convenience function for raising with context
def raise_with_context(exc_class, message, **kwargs):
    """
    Raise an exception with additional context.

    Parameters
    ----------
    exc_class : type
        Exception class to raise.
    message : str
        Error message.
    **kwargs : dict
        Additional context to include in the exception.

    Examples
    --------
    >>> raise_with_context(DataLoadError, "File not found", filepath="data/cwru/122.mat")
    """
    raise exc_class(message, **kwargs)
