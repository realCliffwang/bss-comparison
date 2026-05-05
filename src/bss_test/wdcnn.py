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

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_segments)
    n_classes = len(le.classes_)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_segments)
    y_tensor = torch.LongTensor(y_encoded)

    # Split validation
    n_val = max(1, int(len(X_segments) * validation_split))
    indices = torch.randperm(len(X_segments))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    X_train_split = X_tensor[train_indices]
    y_train_split = y_tensor[train_indices]

    # Create data loaders
    train_dataset = TensorDataset(X_train_split, y_train_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Create model
    model = WDCNN(
        input_length=segment_length,
        n_classes=n_classes,
        **{k: v for k, v in kwargs.items()
           if k in ["filters", "kernel_sizes", "dropout", "n_channels"]}
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    logger.info(f"Training WDCNN on {device}")
    logger.info(f"  Input shape: {X_segments.shape}, Classes: {n_classes}, Epochs: {n_epochs}")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(device)
            y_val_device = y_val.to(device)

            val_outputs = model(X_val_device)
            val_loss = criterion(val_outputs, y_val_device)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == y_val_device).sum().item() / len(y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: "
                       f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                       f"val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}")

    # Attach metadata
    model._label_encoder = le
    model._method = "wdcnn"
    model._device = device

    logger.info(f"Training complete. Final val_acc={history['val_acc'][-1]:.4f}")

    return {"model": model, "history": history}


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

    nn_model = model["model"]
    le = nn_model._label_encoder
    device = nn_model._device

    # Encode labels
    y_encoded = le.transform(y_test)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.LongTensor(y_encoded).to(device)

    # Evaluate
    nn_model.eval()
    with torch.no_grad():
        outputs = nn_model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)

    y_pred_encoded = predicted.cpu().numpy()
    y_pred = le.inverse_transform(y_pred_encoded)

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_encoded, y_pred_encoded)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "true_labels": y_test,
        "label_names": list(le.classes_),
    }
