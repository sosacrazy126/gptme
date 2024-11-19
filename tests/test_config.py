"""Tests for configuration management."""

import os
import pytest
from pathlib import Path
import tempfile
import shutil

from gptme.config import Config, DEFAULT_CONFIG

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def config(temp_dir):
    """Create a test configuration."""
    return Config(data_dir=temp_dir)

def test_default_config(config):
    """Test default configuration values."""
    assert config.model == DEFAULT_CONFIG["model"]
    assert config.temperature == DEFAULT_CONFIG["temperature"]
    assert config.max_tokens == DEFAULT_CONFIG["max_tokens"]
    
    # Test memory defaults
    assert config.memory_enabled == DEFAULT_CONFIG["memory"]["enabled"]
    assert config.memory_storage_type == DEFAULT_CONFIG["memory"]["storage_type"]
    assert config.memory_similarity_threshold == DEFAULT_CONFIG["memory"]["similarity_threshold"]
    assert config.memory_max_context_window == DEFAULT_CONFIG["memory"]["max_context_window"]
    assert config.memory_decay_rate == DEFAULT_CONFIG["memory"]["decay_rate"]

def test_config_override(temp_dir):
    """Test configuration override with constructor arguments."""
    custom_memory = {
        "enabled": False,
        "storage_type": "in_memory",
        "similarity_threshold": 50,
        "max_context_window": 10,
        "decay_rate": 0.001,
    }
    
    config = Config(
        openai_api_key="test-key",
        model="gpt-4",
        temperature=0.5,
        memory_config=custom_memory,
        data_dir=temp_dir,
    )
    
    assert config.openai_api_key == "test-key"
    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    
    # Test memory overrides
    assert config.memory_enabled is False
    assert config.memory_storage_type == "in_memory"
    assert config.memory_similarity_threshold == 50
    assert config.memory_max_context_window == 10
    assert config.memory_decay_rate == 0.001

def test_config_save_load(temp_dir):
    """Test saving and loading configuration."""
    # Create and save config
    config = Config(
        openai_api_key="test-key",
        model="gpt-4",
        memory_config={"enabled": False},
        data_dir=temp_dir,
    )
    config.save()
    
    # Load config and verify values
    new_config = Config(data_dir=temp_dir)
    assert new_config.openai_api_key == "test-key"
    assert new_config.model == "gpt-4"
    assert new_config.memory_enabled is False

def test_env_var_override(temp_dir):
    """Test environment variable override."""
    os.environ["OPENAI_API_KEY"] = "env-test-key"
    
    config = Config(data_dir=temp_dir)
    assert config.openai_api_key == "env-test-key"
    
    # Clean up
    del os.environ["OPENAI_API_KEY"]

def test_memory_property_types(config):
    """Test memory property type conversions."""
    # Test that properties return correct types
    assert isinstance(config.memory_enabled, bool)
    assert isinstance(config.memory_storage_type, str)
    assert isinstance(config.memory_similarity_threshold, float)
    assert isinstance(config.memory_max_context_window, int)
    assert isinstance(config.memory_decay_rate, float)

def test_invalid_config_file(temp_dir):
    """Test handling of invalid configuration file."""
    config_file = Path(temp_dir) / "config.toml"
    
    # Create invalid config file
    with open(config_file, "w") as f:
        f.write("invalid toml content")
    
    # Should fall back to defaults
    config = Config(data_dir=temp_dir)
    assert config.model == DEFAULT_CONFIG["model"]
    assert config.memory_enabled == DEFAULT_CONFIG["memory"]["enabled"]
