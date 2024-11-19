"""Configuration management for gptme."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, NamedTuple

import tomlkit

from .dirs import get_config_dir, get_data_dir

# Global config path
config_path = Path(get_config_dir()) / "config.toml"

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "anthropic_api_key": "",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "memory": {
        "enabled": True,
        "storage_type": "json",  # or "in_memory"
        "similarity_threshold": 40,
        "max_context_window": 5,
        "decay_rate": 0.0001,
    },
    "env": {},  # For environment variables like API keys
    "prompt": {
        "about_user": "You are interacting with a human programmer.",
        "response_preferences": {
            "preferences": []
        },
        "project": {}
    }
}

_config_instance = None

def get_config() -> 'Config':
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def load_config() -> 'Config':
    """Load and return a new configuration instance."""
    return Config()

def set_config_value(key_path: str, value: Any) -> None:
    """Set a configuration value using a dot-notation path.
    
    Args:
        key_path: Dot-notation path to the config value (e.g., 'env.OPENAI_API_KEY')
        value: Value to set
    """
    config = get_config()
    
    # Split the path into parts
    parts = key_path.split('.')
    
    # Navigate to the correct nested dictionary
    current = config.config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Set the value
    current[parts[-1]] = value
    
    # Save the updated config
    config.save()

class ProjectConfig(NamedTuple):
    """Project-specific configuration."""
    files: list[str]

def get_project_config(workspace: Path) -> Optional[ProjectConfig]:
    """Get project-specific configuration from a workspace directory.
    
    Args:
        workspace: Path to the workspace directory
        
    Returns:
        ProjectConfig if a valid configuration is found, None otherwise
    """
    config_file = workspace / "gptme.toml"
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, "r") as f:
            config = tomlkit.load(f)
        
        # Extract project-specific files
        files = config.get("files", [])
        if isinstance(files, list):
            return ProjectConfig(files=files)
    except Exception as e:
        print(f"Error loading project config file: {e}")
    
    return None

class Config:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        data_dir: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ):
        self.config_file = config_path
        self.data_dir = Path(data_dir) if data_dir else Path(get_data_dir())
        
        # Load config from file
        self.config = self._load_config()
        
        # Override with environment variables and constructor arguments
        self.openai_api_key = (
            openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or self.config.get("openai_api_key", "")
        )
        self.anthropic_api_key = (
            anthropic_api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or self.config.get("anthropic_api_key", "")
        )
        self.model = model or self.config.get("model", DEFAULT_CONFIG["model"])
        self.temperature = temperature or self.config.get("temperature", DEFAULT_CONFIG["temperature"])
        self.max_tokens = max_tokens or self.config.get("max_tokens", DEFAULT_CONFIG["max_tokens"])
        
        # Memory configuration
        memory_defaults = DEFAULT_CONFIG["memory"]
        config_memory = self.config.get("memory", {})
        self.memory = {
            **memory_defaults,
            **(config_memory if isinstance(config_memory, dict) else {}),
            **(memory_config or {}),
        }
        
        # Prompt configuration
        self.prompt = self.config.get("prompt", DEFAULT_CONFIG["prompt"])
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_file, "r") as f:
                return dict(tomlkit.load(f))
        except Exception as e:
            print(f"Error loading config file: {e}")
            return DEFAULT_CONFIG.copy()
    
    def save(self) -> None:
        """Save current configuration to file."""
        config_dict = {
            "openai_api_key": self.openai_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "memory": self.memory,
            "env": self.config.get("env", {}),  # Preserve env settings
            "prompt": self.prompt,  # Save prompt settings
        }
        
        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.config_file, "w") as f:
            tomlkit.dump(config_dict, f)
    
    def get_env(self, key: str) -> Optional[str]:
        """Get an environment variable from config or system environment."""
        return self.config.get("env", {}).get(key) or os.getenv(key)
    
    @property
    def memory_enabled(self) -> bool:
        """Check if memory management is enabled."""
        return bool(self.memory.get("enabled", True))
    
    @property
    def memory_storage_type(self) -> str:
        """Get the memory storage type."""
        return str(self.memory.get("storage_type", "json"))
    
    @property
    def memory_similarity_threshold(self) -> float:
        """Get the similarity threshold for memory retrieval."""
        return float(self.memory.get("similarity_threshold", 40))
    
    @property
    def memory_max_context_window(self) -> int:
        """Get the maximum number of context messages to include."""
        return int(self.memory.get("max_context_window", 5))
    
    @property
    def memory_decay_rate(self) -> float:
        """Get the decay rate for memory relevance."""
        return float(self.memory.get("decay_rate", 0.0001))
