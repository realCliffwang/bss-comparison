"""
Shared PyTorch training loop for DL classifiers and WDCNN.

Eliminates code duplication between dl_classifier.py and wdcnn.py.
"""

from typing import Dict, Optional

import numpy as np

from bss_test.utils.logger import get_logger

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
            "PyTorch is required. Install with: pip install torch>=2.0.0"
        )


def train_torch_model(
    model: "nn.Module",
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    device: Optional[str] = None,
    model_name: str = "model",
) -> dict:
    """
    Shared PyTorch training loop with validation split and history tracking.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    X_train : ndarray (n_samples, n_features)
        Training features.
    y_train : ndarray (n_samples,)
        Encoded integer labels.
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
    model_name : str
        Name for logging.

    Returns
    -------
    dict
        {"model": model, "history": history_dict}
    """
    _check_torch()

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)

    # Split validation with safety: at least 1 sample
    n_val = max(1, int(len(X_train) * validation_split))
    indices = torch.randperm(len(X_train))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    X_train_split = X_tensor[train_indices]
    y_train_split = y_tensor[train_indices]

    train_dataset = TensorDataset(X_train_split, y_train_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    logger.info(f"Training {model_name} on {device}")
    logger.info(f"  Input shape: {X_train.shape}, Epochs: {n_epochs}")

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
            logger.info(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}"
            )

    logger.info(f"Training complete. Final val_acc={history['val_acc'][-1]:.4f}")

    return {"model": model, "history": history}


def evaluate_torch_model(
    result: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Evaluate a trained PyTorch model.

    Parameters
    ----------
    result : dict
        {"model": nn.Module, ...} from train_torch_model.
    X_test : ndarray (n_samples, n_features)
    y_test : ndarray (n_samples,)
        Original string labels.

    Returns
    -------
    dict
        {accuracy, f1_macro, confusion_matrix, predictions, true_labels, label_names}
    """
    _check_torch()

    nn_model = result["model"]
    le = nn_model._label_encoder
    device = nn_model._device

    y_encoded = le.transform(y_test)

    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.LongTensor(y_encoded).to(device)

    nn_model.eval()
    with torch.no_grad():
        outputs = nn_model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)

    y_pred_encoded = predicted.cpu().numpy()
    y_pred = le.inverse_transform(y_pred_encoded)

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
