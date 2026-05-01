"""
Tests for configuration module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    ExperimentConfig,
    PreprocessConfig,
    CWTConfig,
    BSSConfig,
    get_config,
    load_config,
    save_config,
    merge_configs,
    dict_to_config,
    config_to_dict,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.dataset == "cwru"
        assert config.data_dir == "data/cwru"
        assert config.fault_type == "inner_race_007"
        assert config.preprocess.detrend is True
        assert config.bss.method == "SOBI"

    def test_sub_configs(self):
        """Test sub-configuration initialization."""
        config = ExperimentConfig()

        assert isinstance(config.preprocess, PreprocessConfig)
        assert isinstance(config.cwt, CWTConfig)
        assert isinstance(config.bss, BSSConfig)

    def test_feature_freqs(self):
        """Test feature frequencies."""
        config = ExperimentConfig()

        assert "BPFO" in config.feature_freqs
        assert "BPFI" in config.feature_freqs
        assert "BSF" in config.feature_freqs


class TestGetConfig:
    """Tests for get_config function."""

    def test_cwru_config(self):
        """Test CWRU configuration."""
        config = get_config("cwru")

        assert config.dataset == "cwru"
        assert "cwru" in config.data_dir

    def test_phm2010_config(self):
        """Test PHM 2010 configuration."""
        config = get_config("phm2010")

        assert config.dataset == "phm2010"
        assert "phm" in config.data_dir

    def test_nasa_config(self):
        """Test NASA configuration."""
        config = get_config("nasa")

        assert config.dataset == "nasa"

    def test_invalid_dataset(self):
        """Test invalid dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_config("invalid")


class TestConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_dict_conversion(self):
        """Test conversion to and from dictionary."""
        config = get_config("cwru")
        data = config_to_dict(config)
        config2 = dict_to_config(data)

        assert config2.dataset == config.dataset
        assert config2.bss.method == config.bss.method

    def test_json_save_load(self, tmp_path):
        """Test saving and loading JSON config."""
        config = get_config("cwru")
        filepath = tmp_path / "config.json"

        save_config(config, str(filepath))
        loaded = load_config(str(filepath))

        assert loaded.dataset == config.dataset
        assert loaded.bss.method == config.bss.method

    def test_yaml_save_load(self, tmp_path):
        """Test saving and loading YAML config."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        config = get_config("cwru")
        filepath = tmp_path / "config.yaml"

        save_config(config, str(filepath))
        loaded = load_config(str(filepath))

        assert loaded.dataset == config.dataset

    def test_load_nonexistent(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.json")

    def test_load_unsupported_format(self, tmp_path):
        """Test loading unsupported format raises error."""
        filepath = tmp_path / "config.txt"
        filepath.write_text("test")

        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(str(filepath))


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_simple_merge(self):
        """Test simple config merge."""
        base = get_config("cwru")
        override = {"fault_type": "ball_007", "bss": {"method": "FastICA"}}

        merged = merge_configs(base, override)

        assert merged.fault_type == "ball_007"
        assert merged.bss.method == "FastICA"
        # Other values should remain
        assert merged.preprocess.detrend == base.preprocess.detrend

    def test_nested_merge(self):
        """Test nested config merge."""
        base = get_config("cwru")
        override = {"bss": {"n_sources": 10}}

        merged = merge_configs(base, override)

        assert merged.bss.n_sources == 10
        assert merged.bss.method == base.bss.method  # Unchanged
