"""
Logging configuration for the context manager system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from .helpers import validate_path


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[Union[str, Path]] = None,
    name: str = "context_manager",
    colored_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
        log_file: Optional log file path
        name: Logger name
        colored_output: Whether to use colored output for console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    if colored_output and sys.stdout.isatty():
        formatter = ColoredFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ContextualLogger:
    """
    Logger with context information support.
    """
    
    def __init__(self, name: str, context: Optional[dict] = None):
        """
        Initialize contextual logger.
        
        Args:
            name: Logger name
            context: Initial context information
        """
        self.logger = logging.getLogger(name)
        self.context = context or {}
        
    def add_context(self, **kwargs) -> None:
        """
        Add context information.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
        
    def clear_context(self) -> None:
        """Clear all context information."""
        self.context.clear()
        
    def _format_message(self, message: str) -> str:
        """
        Format message with context information.
        
        Args:
            message: Original message
            
        Returns:
            Formatted message with context
        """
        if not self.context:
            return message
            
        context_str = " | ".join([f"{k}={v}" for k, v in self.context.items()])
        return f"{message} [{context_str}]"
        
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)
        
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)
        
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)
        
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)
        
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), *args, **kwargs)


def create_log_handler(
    handler_type: str = "console",
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
    colored: bool = True
) -> logging.Handler:
    """
    Create a log handler.
    
    Args:
        handler_type: Type of handler ("console", "file", "rotating_file")
        log_file: Path for file handlers
        level: Logging level
        format_string: Log format string
        colored: Whether to use colored output
        
    Returns:
        Configured log handler
        
    Raises:
        ValueError: If handler type is not supported
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    if colored and handler_type == "console" and sys.stdout.isatty():
        formatter = ColoredFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)
    
    # Create handler
    if handler_type == "console":
        handler = logging.StreamHandler(sys.stdout)
    elif handler_type == "file":
        if not log_file:
            raise ValueError("log_file is required for file handler")
        validate_path(log_file, must_exist=False, create_dirs=True)
        handler = logging.FileHandler(log_file, encoding='utf-8')
    elif handler_type == "rotating_file":
        if not log_file:
            raise ValueError("log_file is required for rotating file handler")
        try:
            from logging.handlers import RotatingFileHandler
            validate_path(log_file, must_exist=False, create_dirs=True)
            handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
        except ImportError:
            raise ValueError("RotatingFileHandler not available")
    else:
        raise ValueError(f"Unsupported handler type: {handler_type}")
    
    # Configure handler
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(formatter)
    
    return handler