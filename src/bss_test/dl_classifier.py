"""
Deep Learning classifiers for vibration fault diagnosis.

Supports:
- 1D-CNN for raw signal classification
- LSTM for sequence classification
- Transformer for attention-based classification

Note: Requires PyTorch. Install with:
    pip install torch>=2.0.0

Usage:
    from bss_test.dl_classifier import train_dl_classifier, evaluate_dl_classifier
    model = train_dl_classifier(X_train, y_train, method="cnn")
    metrics = evaluate_dl_classifier(model, X_test, y_test)
"""

from typing import Dict, List, Optional

import numpy as np

from bss_test.utils.logger import get_logger
from bss_test.utils.exceptions import ClassifierError

logger = get_logger(__name__)

# Check if PyTorch is available
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
            "PyTorch is required for deep learning classifiers. "
            "Install with: pip install torch>=2.0.0"
        )


if TORCH_AVAILABLE:

    class CNN1D(nn.Module):
        """1D Convolutional Neural Network for signal classification."""

        def __init__(
            self,
            input_length: int,
            n_classes: int,
            n_channels: int = 1,
            filters: List[int] = None,
            kernel_sizes: List[int] = None,
            dropout: float = 0.5,
        ):
            super().__init__()

            if filters is None:
                filters = [32, 64, 128]
            if kernel_sizes is None:
                kernel_sizes = [7, 5, 3]

            layers = []
            in_channels = n_channels

            for out_channels, kernel_size in zip(filters, kernel_sizes):
                layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ])
                in_channels = out_channels

            self.features = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(filters[-1], 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            x = self.features(x)
            x = self.classifier(x)
            return x

    class LSTMClassifier(nn.Module):
        """LSTM-based classifier for sequence classification."""

        def __init__(
            self,
            input_size: int,
            n_classes: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.5,
            bidirectional: bool = True,
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            fc_input_size = hidden_size * 2 if bidirectional else hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fc_input_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # Add feature dimension
            lstm_out, _ = self.lstm(x)
            # Use last hidden state
            x = lstm_out[:, -1, :]
            x = self.classifier(x)
            return x

    class TransformerClassifier(nn.Module):
        """Transformer-based classifier for signal classification."""

        def __init__(
            self,
            input_size: int,
            n_classes: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.input_projection = nn.Linear(input_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # Add feature dimension

            # Project to d_model dimensions
            x = self.input_projection(x)

            # Add positional encoding
            seq_len = x.size(1)
            x = x + self.pos_encoding[:, :seq_len, :]

            # Transformer encoding
            x = self.transformer(x)

            # Pool and classify
            x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
            x = self.classifier(x)
            return x


def train_dl_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "cnn",
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    device: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Train a deep learning classifier.

    Parameters
    ----------
    X_train : ndarray (n_samples, n_features)
        Training features.
    y_train : ndarray (n_samples,)
        Training labels.
    method : str
        Model architecture: "cnn", "lstm", "transformer"
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    validation_split : float
        Fraction of data to use for validation.
    device : str or None
        Device to use ("cpu" or "cuda"). If None, auto-detects.
    **kwargs :
        Additional model-specific parameters.

    Returns
    -------
    dict
        Trained model and training history.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    ClassifierError
        If training fails.
    """
    _check_torch()

    from sklearn.preprocessing import LabelEncoder
    from bss_test._torch_training import train_torch_model

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    input_size = X_train.shape[1]
    method = method.lower()

    if method == "cnn":
        model = CNN1D(
            input_length=input_size,
            n_classes=n_classes,
            **{k: v for k, v in kwargs.items() if k in ["filters", "kernel_sizes", "dropout"]}
        )
    elif method == "lstm":
        model = LSTMClassifier(
            input_size=1,
            n_classes=n_classes,
            **{k: v for k, v in kwargs.items() if k in ["hidden_size", "num_layers", "dropout"]}
        )
    elif method == "transformer":
        model = TransformerClassifier(
            input_size=1,
            n_classes=n_classes,
            **{k: v for k, v in kwargs.items() if k in ["d_model", "nhead", "num_layers", "dropout"]}
        )
    else:
        raise ClassifierError(f"Unknown DL method: {method}. Use 'cnn', 'lstm', or 'transformer'")

    result = train_torch_model(
        model, X_train, y_encoded,
        n_epochs=n_epochs, batch_size=batch_size,
        learning_rate=learning_rate, validation_split=validation_split,
        device=device, model_name=method.upper(),
    )

    result["model"]._label_encoder = le
    result["model"]._method = method
    result["model"]._device = next(result["model"].parameters()).device
    return result


def evaluate_dl_classifier(
    model: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Evaluate a trained deep learning classifier.

    Parameters
    ----------
    model : dict
        Model dictionary from train_dl_classifier.
    X_test : ndarray (n_samples, n_features)
        Test features.
    y_test : ndarray (n_samples,)
        Test labels.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    _check_torch()
    from bss_test._torch_training import evaluate_torch_model
    return evaluate_torch_model(model, X_test, y_test)
