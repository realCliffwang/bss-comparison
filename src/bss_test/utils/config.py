"""
Configuration management — backward-compatible re-export layer.

Import from here or from config_types / config_io directly.
"""

from bss_test.utils.config_types import (
    PreprocessConfig,
    TFAConfig,
    BSSConfig,
    FeatureConfig,
    ClassifierConfig,
    VisualizationConfig,
    ExperimentConfig,
    CWTConfig,
    DEFAULT_CONFIGS,
)
from bss_test.utils.config_io import (
    get_config,
    load_config,
    dict_to_config,
    config_to_dict,
    save_config,
    merge_configs,
    deep_update,
    parse_args,
    config_from_args,
    apply_env_overrides,
)

__all__ = [
    "PreprocessConfig", "TFAConfig", "BSSConfig", "FeatureConfig",
    "ClassifierConfig", "VisualizationConfig", "ExperimentConfig",
    "CWTConfig", "DEFAULT_CONFIGS",
    "get_config", "load_config", "dict_to_config", "config_to_dict",
    "save_config", "merge_configs", "deep_update",
    "parse_args", "config_from_args", "apply_env_overrides",
]
