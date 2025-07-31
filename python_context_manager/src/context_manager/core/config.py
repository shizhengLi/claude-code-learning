"""
Configuration management for the context manager system.

This module provides configuration management capabilities with support for
environment variables, configuration files, and runtime configuration updates.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging


@dataclass
class ContextManagerConfig:
    """Configuration class for the context manager system."""
    
    # Context settings
    max_tokens: int = 4000
    compression_ratio: float = 0.8
    context_window_size: int = 100
    
    # Memory settings
    short_term_memory_size: int = 100
    medium_term_memory_size: int = 1000
    long_term_memory_size: int = 10000
    
    # Storage paths
    medium_term_memory_path: str = "data/medium_term_memory.pkl"
    long_term_memory_path: str = "data/long_term_memory.json"
    cache_path: str = "data/cache"
    logs_path: str = "logs"
    
    # Tool settings
    tool_timeout: int = 30
    max_concurrent_tools: int = 10
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Development settings
    debug_mode: bool = False
    enable_profiling: bool = False
    enable_metrics: bool = True
    
    # Advanced settings
    enable_compression: bool = True
    enable_prediction: bool = True
    enable_parallel_execution: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
            
        if not 0 < self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be between 0 and 1")
            
        if self.short_term_memory_size <= 0:
            raise ValueError("short_term_memory_size must be positive")
            
        if self.tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
            
        if self.max_concurrent_tools <= 0:
            raise ValueError("max_concurrent_tools must be positive")
            
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")


class ConfigManager:
    """Configuration manager for the context manager system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default paths.
        """
        self.config_path = Path(config_path) if config_path else self._find_config_file()
        self.config = self._load_config()
        self._watchers = []
        
    def _find_config_file(self) -> Path:
        """Find the configuration file in standard locations."""
        possible_paths = [
            Path("config.json"),
            Path("config/config.json"),
            Path.home() / ".context_manager" / "config.json",
            Path("/etc/context_manager/config.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Return default path
        return Path("config.json")
        
    def _load_config(self) -> ContextManagerConfig:
        """Load configuration from file and environment variables."""
        # Start with default configuration
        config_dict = asdict(ContextManagerConfig())
        
        # Load from file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    config_dict.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
                
        # Override with environment variables
        env_mappings = {
            'CONTEXT_MAX_TOKENS': 'max_tokens',
            'CONTEXT_COMPRESSION_RATIO': 'compression_ratio',
            'CONTEXT_SHORT_TERM_SIZE': 'short_term_memory_size',
            'CONTEXT_MEDIUM_TERM_PATH': 'medium_term_memory_path',
            'CONTEXT_LONG_TERM_PATH': 'long_term_memory_path',
            'CONTEXT_TOOL_TIMEOUT': 'tool_timeout',
            'CONTEXT_LOG_LEVEL': 'log_level',
            'CONTEXT_DEBUG_MODE': 'debug_mode',
            'CONTEXT_ENABLE_CACHING': 'enable_caching',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_key in ['max_tokens', 'short_term_memory_size', 'tool_timeout']:
                    config_dict[config_key] = int(env_value)
                elif config_key in ['compression_ratio']:
                    config_dict[config_key] = float(env_value)
                elif config_key in ['debug_mode', 'enable_caching']:
                    config_dict[config_key] = env_value.lower() in ['true', '1', 'yes', 'on']
                else:
                    config_dict[config_key] = env_value
                    
        return ContextManagerConfig(**config_dict)
        
    def get_config(self) -> ContextManagerConfig:
        """Get the current configuration."""
        return self.config
        
    def update_config(self, **kwargs) -> None:
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration key-value pairs to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
                
        # Validate updated configuration
        self.config._validate_config()
        
        # Notify watchers
        self._notify_watchers()
        
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses current config_path.
        """
        save_path = Path(path) if path else self.config_path
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {save_path}: {e}")
            
    def add_watcher(self, callback) -> None:
        """
        Add a configuration change watcher.
        
        Args:
            callback: Function to call when configuration changes.
        """
        self._watchers.append(callback)
        
    def remove_watcher(self, callback) -> None:
        """
        Remove a configuration change watcher.
        
        Args:
            callback: Function to remove from watchers.
        """
        if callback in self._watchers:
            self._watchers.remove(callback)
            
    def _notify_watchers(self) -> None:
        """Notify all watchers of configuration changes."""
        for callback in self._watchers:
            try:
                callback(self.config)
            except Exception as e:
                logging.error(f"Error in config watcher: {e}")
                
    def create_directories(self) -> None:
        """Create necessary directories for the configuration."""
        directories = [
            Path(self.config.medium_term_memory_path).parent,
            Path(self.config.long_term_memory_path).parent,
            Path(self.config.cache_path),
            Path(self.config.logs_path),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache-specific configuration."""
        return {
            'enabled': self.config.enable_caching,
            'max_size': self.config.cache_size,
            'ttl': self.config.cache_ttl,
            'path': self.config.cache_path,
        }
        
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory-specific configuration."""
        return {
            'short_term_size': self.config.short_term_memory_size,
            'medium_term_path': self.config.medium_term_memory_path,
            'long_term_path': self.config.long_term_memory_path,
            'long_term_size': self.config.long_term_memory_size,
        }
        
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool-specific configuration."""
        return {
            'timeout': self.config.tool_timeout,
            'max_concurrent': self.config.max_concurrent_tools,
            'enable_parallel': self.config.enable_parallel_execution,
        }
        
    def get_context_config(self) -> Dict[str, Any]:
        """Get context-specific configuration."""
        return {
            'max_tokens': self.config.max_tokens,
            'compression_ratio': self.config.compression_ratio,
            'window_size': self.config.context_window_size,
            'enable_compression': self.config.enable_compression,
        }
        
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config_path={self.config_path})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return (f"ConfigManager(config_path={self.config_path}, "
                f"config={self.config})")