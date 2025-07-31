"""
Tests for configuration management.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from context_manager.core.config import ConfigManager, ContextManagerConfig


class TestContextManagerConfig:
    """Test cases for ContextManagerConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContextManagerConfig()
        
        assert config.max_tokens == 4000
        assert config.compression_ratio == 0.8
        assert config.short_term_memory_size == 100
        assert config.log_level == "INFO"
        assert config.enable_caching is True
        assert config.debug_mode is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = ContextManagerConfig(
            max_tokens=2000,
            compression_ratio=0.7
        )
        assert config.max_tokens == 2000
        assert config.compression_ratio == 0.7
        
        # Invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ContextManagerConfig(max_tokens=0)
        
        # Invalid compression_ratio
        with pytest.raises(ValueError, match="compression_ratio must be between 0 and 1"):
            ContextManagerConfig(compression_ratio=1.5)
        
        # Invalid log_level
        with pytest.raises(ValueError, match="log_level must be one of"):
            ContextManagerConfig(log_level="INVALID")
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ContextManagerConfig(
            max_tokens=1000,
            compression_ratio=0.6,
            debug_mode=True
        )
        
        # Test dataclass asdict
        config_dict = config.__dict__
        assert config_dict['max_tokens'] == 1000
        assert config_dict['compression_ratio'] == 0.6
        assert config_dict['debug_mode'] is True


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_config_manager_creation(self):
        """Test basic config manager creation."""
        config_manager = ConfigManager()
        
        assert isinstance(config_manager.config, ContextManagerConfig)
        assert config_manager.config_path is not None
    
    def test_config_loading_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_data = {
            "max_tokens": 3000,
            "compression_ratio": 0.7,
            "debug_mode": True,
            "log_level": "DEBUG"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load configuration from file
            config_manager = ConfigManager(temp_path)
            
            assert config_manager.config.max_tokens == 3000
            assert config_manager.config.compression_ratio == 0.7
            assert config_manager.config.debug_mode is True
            assert config_manager.config.log_level == "DEBUG"
        finally:
            os.unlink(temp_path)
    
    def test_config_environment_variables(self):
        """Test environment variable configuration."""
        # Set environment variables
        os.environ['CONTEXT_MAX_TOKENS'] = '5000'
        os.environ['CONTEXT_COMPRESSION_RATIO'] = '0.9'
        os.environ['CONTEXT_DEBUG_MODE'] = 'true'
        os.environ['CONTEXT_LOG_LEVEL'] = 'WARNING'
        
        try:
            config_manager = ConfigManager()
            
            assert config_manager.config.max_tokens == 5000
            assert config_manager.config.compression_ratio == 0.9
            assert config_manager.config.debug_mode is True
            assert config_manager.config.log_level == "WARNING"
        finally:
            # Clean up environment variables
            for key in ['CONTEXT_MAX_TOKENS', 'CONTEXT_COMPRESSION_RATIO', 
                       'CONTEXT_DEBUG_MODE', 'CONTEXT_LOG_LEVEL']:
                if key in os.environ:
                    del os.environ[key]
    
    def test_config_update(self):
        """Test updating configuration values."""
        config_manager = ConfigManager()
        
        original_tokens = config_manager.config.max_tokens
        
        # Update configuration
        config_manager.update_config(max_tokens=6000, debug_mode=True)
        
        assert config_manager.config.max_tokens == 6000
        assert config_manager.config.debug_mode is True
        
        # Invalid update should raise error
        with pytest.raises(ValueError, match="Unknown configuration key"):
            config_manager.update_config(invalid_key="value")
    
    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create and modify configuration
            config_manager = ConfigManager(temp_path)
            config_manager.update_config(max_tokens=7000, enable_caching=False)
            
            # Save configuration
            config_manager.save_config()
            
            # Load configuration in new manager
            new_config_manager = ConfigManager(temp_path)
            
            assert new_config_manager.config.max_tokens == 7000
            assert new_config_manager.config.enable_caching is False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_config_watchers(self):
        """Test configuration change watchers."""
        config_manager = ConfigManager()
        
        callback_called = False
        received_config = None
        
        def config_callback(config):
            nonlocal callback_called, received_config
            callback_called = True
            received_config = config
        
        # Add watcher
        config_manager.add_watcher(config_callback)
        
        # Update configuration
        config_manager.update_config(max_tokens=8000)
        
        assert callback_called is True
        assert received_config.max_tokens == 8000
        
        # Remove watcher
        config_manager.remove_watcher(config_callback)
        callback_called = False
        
        # Update again
        config_manager.update_config(max_tokens=9000)
        
        assert callback_called is False  # Should not be called
    
    def test_create_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            config_manager = ConfigManager(config_path)
            
            # Set paths to be within temp directory
            config_manager.update_config(
                medium_term_memory_path=str(temp_dir / "medium_term.pkl"),
                long_term_memory_path=str(temp_dir / "long_term.json"),
                cache_path=str(temp_dir / "cache"),
                logs_path=str(temp_dir / "logs")
            )
            
            # Create directories
            config_manager.create_directories()
            
            # Check directories were created
            assert (temp_dir / "cache").exists()
            assert (temp_dir / "logs").exists()
    
    def test_get_specific_configs(self):
        """Test getting specific configuration sections."""
        config_manager = ConfigManager()
        
        # Get cache config
        cache_config = config_manager.get_cache_config()
        assert 'enabled' in cache_config
        assert 'max_size' in cache_config
        assert 'ttl' in cache_config
        
        # Get memory config
        memory_config = config_manager.get_memory_config()
        assert 'short_term_size' in memory_config
        assert 'medium_term_path' in memory_config
        
        # Get tool config
        tool_config = config_manager.get_tool_config()
        assert 'timeout' in tool_config
        assert 'max_concurrent' in tool_config
        
        # Get context config
        context_config = config_manager.get_context_config()
        assert 'max_tokens' in context_config
        assert 'compression_ratio' in context_config
    
    def test_config_file_not_found(self):
        """Test behavior when config file doesn't exist."""
        non_existent_path = "/tmp/non_existent_config.json"
        
        # Should not raise error, should use defaults
        config_manager = ConfigManager(non_existent_path)
        
        assert isinstance(config_manager.config, ContextManagerConfig)
        assert config_manager.config.max_tokens == 4000  # Default value
    
    def test_invalid_config_file(self):
        """Test behavior with invalid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            # Should not raise error, should use defaults
            config_manager = ConfigManager(temp_path)
            
            assert isinstance(config_manager.config, ContextManagerConfig)
        finally:
            os.unlink(temp_path)
    
    def test_string_representation(self):
        """Test string representation of config manager."""
        config_manager = ConfigManager()
        
        str_repr = str(config_manager)
        assert "ConfigManager" in str_repr
        assert "config_path" in str_repr
        
        repr_str = repr(config_manager)
        assert "ConfigManager" in repr_str
        assert "config_path" in repr_str