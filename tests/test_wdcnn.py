"""
Tests for WDCNN module.
"""

from pathlib import Path

import numpy as np
import pytest

# Skip all tests if PyTorch not installed
torch = pytest.importorskip("torch")

from bss_test.wdcnn import WDCNN, segment_signals, train_wdcnn, evaluate_wdcnn


class TestWDCNNModel:
    """Tests for WDCNN model architecture."""

    def test_output_shape(self):
        """Test forward pass output shape."""
        model = WDCNN(input_length=1024, n_classes=4)
        x = torch.randn(8, 1, 1024)
        out = model(x)
        assert out.shape == (8, 4)

    def test_output_shape_2d_input(self):
        """Test forward pass with 2D input (no channel dim)."""
        model = WDCNN(input_length=1024, n_classes=4)
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 4)

    def test_different_n_classes(self):
        """Test model with different number of classes."""
        for n_classes in [2, 4, 10]:
            model = WDCNN(input_length=1024, n_classes=n_classes)
            x = torch.randn(4, 1, 1024)
            out = model(x)
            assert out.shape == (4, n_classes)

    def test_custom_params(self):
        """Test model with custom filter and kernel parameters."""
        model = WDCNN(
            input_length=1024,
            n_classes=4,
            filters=[32, 16, 32, 64, 64],
            kernel_sizes=[32, 3, 3, 3, 3],
            dropout=0.3,
        )
        x = torch.randn(4, 1, 1024)
        out = model(x)
        assert out.shape == (4, 4)

    def test_model_parameters_exist(self):
        """Test that model has trainable parameters."""
        model = WDCNN(input_length=1024, n_classes=4)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestSegmentSignals:
    """Tests for signal segmentation."""

    def test_basic_segmentation(self):
        """Test basic signal segmentation."""
        signals = np.random.randn(1, 4096)
        labels = np.array(["fault_a"])
        X_seg, y_seg = segment_signals(signals, labels, segment_length=1024, overlap=0.5)

        # With 4096 samples, 1024 length, 50% overlap: ~7 segments
        assert X_seg.shape[1] == 1024
        assert len(X_seg) == len(y_seg)
        assert len(X_seg) > 0

    def test_1d_input(self):
        """Test segmentation with 1D input signal."""
        signal = np.random.randn(2048)
        labels = np.array(["fault"])
        X_seg, y_seg = segment_signals(signal, labels, segment_length=1024, overlap=0.5)

        assert X_seg.shape[1] == 1024
        assert len(X_seg) == len(y_seg)

    def test_overlap_count(self):
        """Test that overlap produces expected number of segments."""
        n_samples = 2048
        segment_length = 1024
        overlap = 0.5
        step = int(segment_length * (1 - overlap))

        signals = np.random.randn(1, n_samples)
        labels = np.array(["a"])
        X_seg, y_seg = segment_signals(signals, labels, segment_length=segment_length, overlap=overlap)

        # Expected: floor((n_samples - segment_length) / step) + 1
        expected = (n_samples - segment_length) // step + 1
        assert len(X_seg) == expected

    def test_signal_too_short(self):
        """Test that short signal raises ValueError."""
        signals = np.random.randn(1, 100)
        labels = np.array(["a"])

        with pytest.raises(ValueError, match="Signal length"):
            segment_signals(signals, labels, segment_length=1024)

    def test_label_propagation(self):
        """Test that labels are correctly propagated to segments."""
        signals = np.random.randn(1, 2048)
        labels = np.array(["inner_race"])
        X_seg, y_seg = segment_signals(signals, labels, segment_length=1024, overlap=0.5)

        assert all(l == "inner_race" for l in y_seg)

    def test_multi_channel_segmentation(self):
        """Test segmentation with multiple channels."""
        signals = np.random.randn(3, 4096)
        labels = np.array(["a", "b", "c"])
        X_seg, y_seg = segment_signals(signals, labels, segment_length=1024, overlap=0.5)

        # Each channel produces segments independently
        assert X_seg.shape[1] == 1024
        assert len(X_seg) == len(y_seg)


class TestTrainWDCNN:
    """Tests for WDCNN training."""

    def test_returns_dict(self):
        """Test that train_wdcnn returns correct dict format."""
        np.random.seed(42)
        X = np.random.randn(40, 1024).astype(np.float32)
        y = np.array(["a"] * 20 + ["b"] * 20)

        result = train_wdcnn(X, y, n_epochs=2, batch_size=8, validation_split=0.2)

        assert "model" in result
        assert "history" in result
        assert "train_loss" in result["history"]
        assert "val_loss" in result["history"]
        assert "train_acc" in result["history"]
        assert "val_acc" in result["history"]

    def test_long_signal_input(self):
        """Test training with long signal that needs segmentation."""
        np.random.seed(42)
        signals = np.random.randn(1, 4096).astype(np.float32)
        labels = np.array(["fault"])

        result = train_wdcnn(
            signals, labels,
            segment_length=1024, overlap=0.5,
            n_epochs=2, batch_size=4, validation_split=0.2,
        )

        assert "model" in result
        assert result["model"]._method == "wdcnn"

    def test_custom_params(self):
        """Test training with custom model parameters."""
        np.random.seed(42)
        X = np.random.randn(40, 1024).astype(np.float32)
        y = np.array(["a"] * 20 + ["b"] * 20)

        result = train_wdcnn(
            X, y, n_epochs=2, batch_size=8,
            filters=[32, 16, 32, 64, 64],
            kernel_sizes=[32, 3, 3, 3, 3],
        )

        assert "model" in result


class TestEvaluateWDCNN:
    """Tests for WDCNN evaluation."""

    def test_metrics_format(self):
        """Test that evaluate_wdcnn returns correct metric format."""
        np.random.seed(42)
        X = np.random.randn(40, 1024).astype(np.float32)
        y = np.array(["a"] * 20 + ["b"] * 20)

        result = train_wdcnn(X, y, n_epochs=2, batch_size=8, validation_split=0.1)
        metrics = evaluate_wdcnn(result, X[:10], y[:10])

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics
        assert "predictions" in metrics
        assert "true_labels" in metrics
        assert "label_names" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_predictions_shape(self):
        """Test that predictions have correct shape."""
        np.random.seed(42)
        X = np.random.randn(40, 1024).astype(np.float32)
        y = np.array(["a"] * 20 + ["b"] * 20)

        result = train_wdcnn(X, y, n_epochs=2, batch_size=8, validation_split=0.1)
        metrics = evaluate_wdcnn(result, X[:10], y[:10])

        assert len(metrics["predictions"]) == 10
        assert len(metrics["true_labels"]) == 10


@pytest.mark.skipif(
    not (Path("data/cwru").exists()),
    reason="CWRU data not available"
)
class TestWDCNNIntegration:
    """Integration tests with real CWRU data."""

    def test_end_to_end_cwru(self):
        """Test WDCNN end-to-end on CWRU data."""
        from pathlib import Path
        from bss_test.io.cwru import load_cwru
        from bss_test.preprocessing import preprocess_signals

        fault_types = ["normal", "inner_race_007", "ball_007", "outer_race_6_007"]
        all_signals = []
        all_labels = []

        for fault_type in fault_types:
            try:
                signals, fs, rpm = load_cwru(
                    data_dir="data/cwru",
                    fault_type=fault_type,
                    load=0,
                    channels=["DE"],
                )
                n_use = min(signals.shape[1], int(1.0 * fs))
                signals = signals[:, :n_use]

                preprocess_config = {
                    "detrend": True,
                    "bandpass": [100, 5000],
                    "normalize": "zscore",
                }
                signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)

                all_signals.append(signals_pre[0])
                all_labels.append(fault_type)
            except Exception:
                pytest.skip(f"CWRU data loading failed for {fault_type}")

        if not all_signals:
            pytest.skip("No CWRU data loaded")

        # Segment signals
        all_X = []
        all_y = []
        for sig, label in zip(all_signals, all_labels):
            X_seg, y_seg = segment_signals(
                sig.reshape(1, -1),
                np.array([label]),
                segment_length=1024,
                overlap=0.5,
            )
            all_X.append(X_seg)
            all_y.append(y_seg)

        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )

        # Train
        result = train_wdcnn(X_train, y_train, n_epochs=10, batch_size=32)

        # Evaluate
        metrics = evaluate_wdcnn(result, X_test, y_test)

        # WDCNN should achieve reasonable accuracy on CWRU
        assert metrics["accuracy"] > 0.5, f"Accuracy {metrics['accuracy']:.4f} too low"
