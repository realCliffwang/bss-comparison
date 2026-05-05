"""
Tests for report generation module.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test.report import ExperimentReport


class TestExperimentReport:
    """Tests for ExperimentReport class."""

    def test_init(self, tmp_path):
        """Test report initialization."""
        report = ExperimentReport("Test Report", str(tmp_path))
        assert report.title == "Test Report"
        assert report.output_dir.exists()

    def test_add_text(self, tmp_path):
        """Test adding text sections."""
        report = ExperimentReport("Test", str(tmp_path))
        report.add_text("Setup", "This is the setup description.")
        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "text"

    def test_add_figure(self, tmp_path):
        """Test adding figures."""
        report = ExperimentReport("Test", str(tmp_path))
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        report.add_figure(fig, "Test Plot")
        plt.close(fig)
        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "figure"

    def test_add_table(self, tmp_path):
        """Test adding tables."""
        report = ExperimentReport("Test", str(tmp_path))
        report.add_table(
            ["Method", "Accuracy"],
            [["SVM", "0.95"], ["RF", "0.92"]],
            caption="Results",
        )
        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "table"

    def test_add_metrics_table(self, tmp_path):
        """Test adding metrics table from results."""
        report = ExperimentReport("Test", str(tmp_path))
        results = [
            {"method": "svm", "accuracy": 0.95, "f1_macro": 0.94, "time": 1.5},
            {"method": "rf", "accuracy": 0.92, "f1_macro": 0.91, "time": 2.3},
        ]
        report.add_metrics_table(results, "ML Results")
        assert len(report.sections) == 1
        section = report.sections[0]
        assert section["headers"] == ["Method", "Accuracy", "F1-Macro", "Time (s)"]

    def test_to_html(self, tmp_path):
        """Test HTML report generation."""
        report = ExperimentReport("Test Report", str(tmp_path))
        report.set_metadata("dataset", "CWRU")
        report.add_text("Setup", "Test setup description.")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        report.add_figure(fig, "Test Plot")
        plt.close(fig)

        results = [{"method": "svm", "accuracy": 0.95, "f1_macro": 0.94}]
        report.add_metrics_table(results, "Results")

        html = report.to_html()
        filepath = tmp_path / "report.html"
        assert filepath.exists()
        assert "Test Report" in html
        assert "CWRU" in html
        assert "data:image/png;base64" in html

    def test_to_markdown(self, tmp_path):
        """Test Markdown report generation."""
        report = ExperimentReport("Test Report", str(tmp_path))
        report.set_metadata("dataset", "CWRU")
        report.add_text("Setup", "Test setup description.")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        report.add_figure(fig, "Test Plot")
        plt.close(fig)

        results = [{"method": "svm", "accuracy": 0.95, "f1_macro": 0.94}]
        report.add_metrics_table(results, "Results")

        md = report.to_markdown()
        filepath = tmp_path / "report.md"
        assert filepath.exists()
        assert "# Test Report" in md
        assert "CWRU" in md
        assert "![Test Plot]" in md
        assert "| SVM |" in md

    def test_auto_filename(self, tmp_path):
        """Test auto-generated figure filename."""
        report = ExperimentReport("Test", str(tmp_path))
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        report.add_figure(fig, "My Cool Plot")
        plt.close(fig)

        assert report.sections[0]["filename"] == "my_cool_plot.png"

    def test_html_escaping(self, tmp_path):
        """Test that HTML special characters are escaped in output."""
        report = ExperimentReport("Test", str(tmp_path))
        report.add_text("Section <script>alert('xss')</script>", "Content with <b>tags</b>")
        html = report.to_html()

        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;b&gt;" in html
