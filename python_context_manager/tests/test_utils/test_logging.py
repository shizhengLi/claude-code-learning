"""
Tests for logging utilities.
"""

import pytest
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from context_manager.utils.logging import (
    ColoredFormatter, setup_logging, get_logger, ContextualLogger,
    create_log_handler
)


class TestColoredFormatter:
    """Test cases for ColoredFormatter class."""
    
    def test_colored_formatter_creation(self):
        """Test basic colored formatter creation."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        
        assert formatter.COLORS is not None
        assert 'DEBUG' in formatter.COLORS
        assert 'INFO' in formatter.COLORS
        assert 'ERROR' in formatter.COLORS
    
    def test_colored_formatter_format(self):
        """Test colored formatter formatting."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should contain color codes
        assert '\033[' in formatted
        assert 'Test message' in formatted
        assert 'INFO' in formatted
    
    def test_colored_formatter_unknown_level(self):
        """Test colored formatter with unknown level."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        
        record = logging.LogRecord(
            name="test_logger",
            level=99,  # Unknown level
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Should not raise error
        formatted = formatter.format(record)
        assert 'Test message' in formatted


class TestSetupLogging:
    """Test cases for setup_logging function."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "context_manager"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1  # At least console handler
    
    def test_setup_logging_with_custom_level(self):
        """Test logging setup with custom level."""
        logger = setup_logging(log_level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_with_custom_name(self):
        """Test logging setup with custom name."""
        logger = setup_logging(name="custom_logger")
        
        assert logger.name == "custom_logger"
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            logger = setup_logging(log_file=log_file)
            
            # Should have both console and file handlers
            assert len(logger.handlers) >= 2
            
            # Check file handler exists
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) >= 1
    
    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging clears existing handlers."""
        # Create logger with existing handlers
        logger = logging.getLogger("test_clear_handlers")
        existing_handler = logging.StreamHandler()
        logger.addHandler(existing_handler)
        
        # Setup logging should clear existing handlers
        new_logger = setup_logging(name="test_clear_handlers")
        
        assert len(new_logger.handlers) >= 1
        assert existing_handler not in new_logger.handlers
    
    def test_setup_logging_colored_output(self):
        """Test colored output configuration."""
        logger = setup_logging(colored_output=True)
        
        # Check if console handler uses colored formatter
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        if console_handlers and sys.stdout.isatty():
            formatter = console_handlers[0].formatter
            assert isinstance(formatter, ColoredFormatter)
    
    def test_setup_logging_no_colored_output(self):
        """Test no colored output configuration."""
        logger = setup_logging(colored_output=False)
        
        # Check if console handler uses regular formatter
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1
        
        formatter = console_handlers[0].formatter
        assert not isinstance(formatter, ColoredFormatter)


class TestGetLogger:
    """Test cases for get_logger function."""
    
    def test_get_logger_basic(self):
        """Test basic logger retrieval."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_same_instance(self):
        """Test that get_logger returns same instance for same name."""
        logger1 = get_logger("same_logger")
        logger2 = get_logger("same_logger")
        
        assert logger1 is logger2


class TestContextualLogger:
    """Test cases for ContextualLogger class."""
    
    def test_contextual_logger_creation(self):
        """Test basic contextual logger creation."""
        logger = ContextualLogger("test_logger")
        
        assert isinstance(logger.logger, logging.Logger)
        assert logger.context == {}
    
    def test_contextual_logger_with_initial_context(self):
        """Test contextual logger with initial context."""
        initial_context = {"user_id": "123", "session": "abc"}
        logger = ContextualLogger("test_logger", context=initial_context)
        
        assert logger.context == initial_context
    
    def test_add_context(self):
        """Test adding context information."""
        logger = ContextualLogger("test_logger")
        
        logger.add_context(user_id="123", operation="test")
        
        assert logger.context == {"user_id": "123", "operation": "test"}
    
    def test_add_context_to_existing(self):
        """Test adding context to existing context."""
        logger = ContextualLogger("test_logger", context={"user_id": "123"})
        
        logger.add_context(operation="test")
        
        assert logger.context == {"user_id": "123", "operation": "test"}
    
    def test_clear_context(self):
        """Test clearing context information."""
        logger = ContextualLogger("test_logger", context={"user_id": "123"})
        
        logger.clear_context()
        
        assert logger.context == {}
    
    def test_format_message_with_context(self):
        """Test message formatting with context."""
        logger = ContextualLogger("test_logger", context={"user_id": "123"})
        
        formatted = logger._format_message("Test message")
        
        assert "Test message" in formatted
        assert "user_id=123" in formatted
        assert "[" in formatted and "]" in formatted
    
    def test_format_message_without_context(self):
        """Test message formatting without context."""
        logger = ContextualLogger("test_logger")
        
        formatted = logger._format_message("Test message")
        
        assert formatted == "Test message"
    
    def test_debug_logging_with_context(self):
        """Test debug logging with context."""
        logger = ContextualLogger("test_logger", context={"user_id": "123"})
        
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.debug("Test message")
            
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0]
            assert "Test message" in call_args[0]
            assert "user_id=123" in call_args[0]
    
    def test_info_logging_with_context(self):
        """Test info logging with context."""
        logger = ContextualLogger("test_logger", context={"operation": "test"})
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Test message")
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0]
            assert "Test message" in call_args[0]
            assert "operation=test" in call_args[0]
    
    def test_warning_logging_with_context(self):
        """Test warning logging with context."""
        logger = ContextualLogger("test_logger", context={"session": "abc"})
        
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.warning("Test message")
            
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0]
            assert "Test message" in call_args[0]
            assert "session=abc" in call_args[0]
    
    def test_error_logging_with_context(self):
        """Test error logging with context."""
        logger = ContextualLogger("test_logger", context={"error_code": "500"})
        
        with patch.object(logger.logger, 'error') as mock_error:
            logger.error("Test message")
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0]
            assert "Test message" in call_args[0]
            assert "error_code=500" in call_args[0]
    
    def test_critical_logging_with_context(self):
        """Test critical logging with context."""
        logger = ContextualLogger("test_logger", context={"severity": "high"})
        
        with patch.object(logger.logger, 'critical') as mock_critical:
            logger.critical("Test message")
            
            mock_critical.assert_called_once()
            call_args = mock_critical.call_args[0]
            assert "Test message" in call_args[0]
            assert "severity=high" in call_args[0]
    
    def test_logging_with_args_and_kwargs(self):
        """Test logging with format args and kwargs."""
        logger = ContextualLogger("test_logger", context={"user_id": "123"})
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("User %s has %d items", "Alice", 5, extra={"custom": "value"})
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0] == ("User %s has %d items [user_id=123]", "Alice", 5)
            assert call_args[1]['extra'] == {"custom": "value"}


class TestCreateLogHandler:
    """Test cases for create_log_handler function."""
    
    def test_create_console_handler(self):
        """Test creating console handler."""
        handler = create_log_handler(handler_type="console")
        
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.INFO
        assert handler.formatter is not None
    
    def test_create_console_handler_with_level(self):
        """Test creating console handler with custom level."""
        handler = create_log_handler(handler_type="console", level="DEBUG")
        
        assert handler.level == logging.DEBUG
    
    def test_create_console_handler_colored(self):
        """Test creating colored console handler."""
        handler = create_log_handler(handler_type="console", colored=True)
        
        if sys.stdout.isatty():
            assert isinstance(handler.formatter, ColoredFormatter)
    
    def test_create_console_handler_not_colored(self):
        """Test creating non-colored console handler."""
        handler = create_log_handler(handler_type="console", colored=False)
        
        assert not isinstance(handler.formatter, ColoredFormatter)
    
    def test_create_file_handler(self):
        """Test creating file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            handler = create_log_handler(handler_type="file", log_file=log_file)
            
            assert isinstance(handler, logging.FileHandler)
            assert handler.baseFilename == str(log_file)
    
    def test_create_file_handler_without_path(self):
        """Test creating file handler without path should raise error."""
        with pytest.raises(ValueError, match="log_file is required"):
            create_log_handler(handler_type="file")
    
    def test_create_rotating_file_handler(self):
        """Test creating rotating file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            handler = create_log_handler(handler_type="rotating_file", log_file=log_file)
            
            assert hasattr(handler, 'maxBytes')
            assert hasattr(handler, 'backupCount')
    
    def test_create_rotating_file_handler_without_path(self):
        """Test creating rotating file handler without path should raise error."""
        with pytest.raises(ValueError, match="log_file is required"):
            create_log_handler(handler_type="rotating_file")
    
    def test_create_handler_custom_format(self):
        """Test creating handler with custom format."""
        custom_format = "%(name)s - %(message)s"
        handler = create_log_handler(handler_type="console", format_string=custom_format)
        
        assert custom_format in handler.formatter._fmt
    
    def test_create_handler_unsupported_type(self):
        """Test creating handler with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported handler type"):
            create_log_handler(handler_type="unsupported_type")
    
    def test_file_handler_creates_directories(self):
        """Test that file handler creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "logs" / "subdir" / "test.log"
            
            # Directory should not exist initially
            assert not log_file.parent.exists()
            
            handler = create_log_handler(handler_type="file", log_file=log_file)
            
            # Directory should be created
            assert log_file.parent.exists()


# Import tempfile for tests
import tempfile