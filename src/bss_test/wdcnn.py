"""
WDCNN (Wide Deep CNN) for raw vibration signal fault diagnosis.

Reference:
  Zhang, W., Peng, G., Li, C., Chen, Y., & Zhang, Z. (2017).
  A new deep learning model for fault diagnosis with good anti-noise
  and domain adaptation ability on raw vibration signals.
  Sensors, 17(2), 425.

Architecture:
  Wide first conv layer (kernel=64, stride=16) captures low-frequency
 冲击 features. Deep subsequent layers (kernel=3) extract high-level
  semantic features. BatchNorm + Dropout for regularization.

Note: Requires PyTorch. Install with:
    pip install torch>=2.0.0

Usage:
    from bss_test.wdcnn import train_wdcnn, evaluate_wdcnn, segment_signals

    # From raw signals
    X_seg, y_seg = segment_signals(signals, labels)
    result = train_wdcnn(X_seg, y_seg)
    metrics = evaluate_wdcnn(result, X_test, y_test)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from bss_test.utils.logger import get_logger
from bss_test.utils.exceptions import ClassifierError

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for WDCNN. "
            "Install with: pip install torch>=2.0.0"
        )


if TORCH_AVAILABLE:

    class WDCNN(nn.Module):
        """Wide Deep CNN for raw 1D vibration signal classification."""

        def __init__(
            self,
            input_length: int = 1024,
            n_classes: int = 4,
            n_channels: int = 1,
            dropout: float = 0.5,
            filters: List[int] = None,
            kernel_sizes: List[int] = None,
        ):
            super().__init__()

            if filters is None:
                filters = [64, 32, 64, 128, 128]
            if kernel_sizes is None:
                kernel_sizes = [64, 3, 3, 3, 3]

            # Wide first layer
            self.wide = nn.Sequential(
                nn.Conv1d(n_channels, filters[0], kernel_sizes[0], stride=16),
                nn.BatchNorm1d(filters[0]),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

            # Deep layers
            deep_layers = []
            in_channels = filters[0]
            for out_channels, kernel_size in zip(filters[1:], kernel_sizes[1:]):
                deep_layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ])
                in_channels = out_channels
            self.deep = nn.Sequential(*deep_layers)

            # Classifier head
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(filters[-1], n_classes),
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            x = self.wide(x)
            x = self.deep(x)
            x = self.classifier(x)
            return x


def segment_signals(
    signals: np.ndarray,
    labels: np.ndarray,
    segment_length: int = 1024,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment long signals into fixed-length windows with overlap.

    Parameters
    ----------
    signals : ndarray (n_channels, n_samples) or (n_samples,)
        Raw vibration signal(s).
    labels : ndarray
        Label for each signal (or single label if 1D input).
    segment_length : int
        Length of each segment in samples.
    overlap : float
        Overlap ratio between consecutive segments (0 to 1).

    Returns
    -------
    X_segments : ndarray (n_segments, segment_length)
        Segmented signals.
    y_segments : ndarray (n_segments,)
        Labels for each segment.

    Raises
    ------
    ValueError
        If signal length is less than segment_length.
    """
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_channels, n_samples = signals.shape

    if n_samples < segment_length:
        raise ValueError(
            f"Signal length ({n_samples}) must be >= segment_length ({segment_length})"
        )

    step = int(segment_length * (1 - overlap))
    if step < 1:
        step = 1

    segments = []
    seg_labels = []

    for ch in range(n_channels):
        sig = signals[ch]
        label = labels[ch] if len(labels) > 1 else labels[0]

        start = 0
        while start + segment_length <= n_samples:
            segments.append(sig[start:start + segment_length])
            seg_labels.append(label)
            start += step

    return np.array(segments), np.array(seg_labels)


def train_wdcnn(
    signals: np.ndarray,
    labels: np.ndarray,
    segment_length: int = 1024,
    overlap: float = 0.5,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    device: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Train a WDCNN classifier on raw vibration signals.

    Parameters
    ----------
    signals : ndarray
        Raw signals. Shape (n_segments, segment_length) for pre-segmented
        data, or (n_channels, n_samples) for long signals that will be
        segmented internally.
    labels : ndarray
        Labels corresponding to signals.
    segment_length : int
        Segment length for signal segmentation (used only if input is long signal).
    overlap : float
        Overlap ratio (used only if input is long signal).
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    validation_split : float
        Fraction of data for validation.
    device : str or None
        Device ("cpu" or "cuda"). If None, auto-detects.
    **kwargs :
        Additional model parameters (filters, kernel_sizes, dropout).

    Returns
    -------
    dict
        {"model": model_dict, "history": history_dict}

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    ValueError
        If labels and signals have mismatched counts.
    """
    _check_torch()

    from sklearn.preprocessing import LabelEncoder
    from bss_test._torch_training import train_torch_model

    # Auto-detect input format: long signal vs pre-segmented
    if signals.ndim == 1 or (signals.ndim == 2 and signals.shape[0] < signals.shape[1]):
        logger.info("Input detected as long signal, segmenting...")
        X_segments, y_segments = segment_signals(
            signals, labels, segment_length=segment_length, overlap=overlap
        )
    else:
        X_segments = signals
        y_segments = labels

    if len(X_segments) != len(y_segments):
        raise ValueError(
            f"Signal segments ({len(X_segments)}) and labels ({len(y_segments)}) count mismatch"
        )

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_segments)
    n_classes = len(le.classes_)

    model = WDCNN(
        input_length=segment_length,
        n_classes=n_classes,
        **{k: v for k, v in kwargs.items()
           if k in ["filters", "kernel_sizes", "dropout", "n_channels"]}
    )

    result = train_torch_model(
        model, X_segments, y_encoded,
        n_epochs=n_epochs, batch_size=batch_size,
        learning_rate=learning_rate, validation_split=validation_split,
        device=device, model_name="WDCNN",
    )

    result["model"]._label_encoder = le
    result["model"]._method = "wdcnn"
    result["model"]._device = next(result["model"].parameters()).device
    return result


def evaluate_wdcnn(
    model: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Evaluate a trained WDCNN classifier.

    Parameters
    ----------
    model : dict
        Model dictionary from train_wdcnn.
    X_test : ndarray (n_samples, segment_length)
        Test signal segments.
    y_test : ndarray (n_samples,)
        Test labels.

    Returns
    -------
    dict
        {"accuracy", "f1_macro", "confusion_matrix",
         "predictions", "true_labels", "label_names"}
    """
    _check_torch()
    from bss_test._torch_training import evaluate_torch_model
    return evaluate_torch_model(model, X_test, y_test)
