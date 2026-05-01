"""
BSS-Test: 轴承故障诊断盲源分离框架
"""

__version__ = "0.3.0"
__author__ = "BSS-Test Contributors"

# 常用导入快捷方式
from bss_test.preprocessing import preprocess_signals
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
)
from bss_test.utils.config import ExperimentConfig
from bss_test.utils.logger import get_logger, setup_logging
from bss_test.utils.exceptions import (
    BSSTestError,
    DataLoadError,
    PreprocessingError,
    TFAError,
    BSSError,
    FeatureExtractionError,
    ClassifierError,
    ConfigurationError,
)

# 深度学习分类器（可选依赖 PyTorch）
try:
    from bss_test.dl_classifier import train_dl_classifier, evaluate_dl_classifier
except ImportError:
    pass

# 版本信息
VERSION_INFO = {
    "version": __version__,
    "major": 0,
    "minor": 3,
    "patch": 0,
}


def get_version() -> str:
    """获取当前版本字符串。"""
    return __version__


def get_version_info() -> dict:
    """获取详细版本信息。"""
    return VERSION_INFO.copy()
