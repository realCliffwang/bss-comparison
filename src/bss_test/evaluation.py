"""
Evaluation and visualization for BSS results.

This module re-exports from metrics.py and visualization.py for backward
compatibility. All existing imports continue to work:

    from bss_test.evaluation import compute_independence_metric
    from bss_test.evaluation import plot_envelope_spectrum
    from bss_test.evaluation import setup_academic_style
"""

from bss_test.metrics import (
    compute_metrics,
    compute_independence_metric,
    compute_fault_detection_score,
    evaluate_bss,
)

from bss_test.visualization import (
    ACADEMIC_STYLE,
    setup_academic_style,
    plot_waveform_comparison,
    plot_spectrum_comparison,
    plot_envelope_spectrum,
    plot_correlation_matrix,
    plot_wear_evolution,
    plot_bss_metrics_comparison,
    plot_tfa_metrics_comparison,
    plot_tfa_comparison,
    plot_bss_comparison,
    plot_classifier_comparison,
    plot_confusion_matrix_grid,
    plot_tfa_bss_cross_comparison,
    plot_separation_quality_report,
)

__all__ = [
    # Metrics
    "compute_metrics",
    "compute_independence_metric",
    "compute_fault_detection_score",
    # Visualization
    "ACADEMIC_STYLE",
    "setup_academic_style",
    "evaluate_bss",
    "plot_waveform_comparison",
    "plot_spectrum_comparison",
    "plot_envelope_spectrum",
    "plot_correlation_matrix",
    "plot_wear_evolution",
    "plot_bss_metrics_comparison",
    "plot_tfa_metrics_comparison",
    "plot_tfa_comparison",
    "plot_bss_comparison",
    "plot_classifier_comparison",
    "plot_confusion_matrix_grid",
    "plot_tfa_bss_cross_comparison",
    "plot_separation_quality_report",
]
