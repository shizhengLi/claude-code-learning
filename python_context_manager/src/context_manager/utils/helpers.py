"""
Utility components for the context manager system.

This module provides logging, error handling, and helper functions
used throughout the system.
"""

import logging
import sys
import time
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from pathlib import Path
import asyncio
import json
import platform
import psutil


# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ContextManagerError(Exception):
    """Base exception for context manager errors."""
    pass


class MemoryError(ContextManagerError):
    """Exception raised for memory-related errors."""
    pass


class ToolError(ContextManagerError):
    """Exception raised for tool-related errors."""
    pass


class ConfigurationError(ContextManagerError):
    """Exception raised for configuration-related errors."""
    pass


class StorageError(ContextManagerError):
    """Exception raised for storage-related errors."""
    pass


class CompressionError(ContextManagerError):
    """Exception raised for compression-related errors."""
    pass


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[Union[str, Path]] = None,
    name: str = "context_manager"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_format: Log message format
        log_file: Optional log file path
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
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
        file_handler.setFormatter(formatter)
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


class Timer:
    """
    Context manager for timing operations.
    
    Usage:
        with Timer("operation") as timer:
            # Do something
        print(f"Operation took {timer.elapsed:.2f} seconds")
    """
    
    def __init__(self, name: str = "operation", logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
            logger: Optional logger for timing information
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0
        
    def __enter__(self) -> 'Timer':
        """Start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and log result."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        self.logger.info(f"{self.name} took {self.elapsed:.4f} seconds")
        
    def __str__(self) -> str:
        """String representation of timing result."""
        return f"{self.name}: {self.elapsed:.4f}s"


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry on
        logger: Optional logger for retry information
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        if logger:
                            func_name = getattr(func, '__name__', 'unknown_function')
                            logger.error(
                                f"Function {func_name} failed after {max_attempts} attempts. "
                                f"Last error: {e}"
                            )
                        raise e
                    
                    if logger:
                        func_name = getattr(func, '__name__', 'unknown_function')
                        logger.warning(
                            f"Function {func_name} failed on attempt {attempt + 1}/{max_attempts}. "
                            f"Retrying in {current_delay:.2f}s. Error: {e}"
                        )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but just in case
            raise last_exception if last_exception else RuntimeError("Retry decorator failed")
        
        return wrapper
    
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying async functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry on
        logger: Optional logger for retry information
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        if logger:
                            func_name = getattr(func, '__name__', 'unknown_async_function')
                            logger.error(
                                f"Async function {func_name} failed after {max_attempts} attempts. "
                                f"Last error: {e}"
                            )
                        raise e
                    
                    if logger:
                        func_name = getattr(func, '__name__', 'unknown_async_function')
                        logger.warning(
                            f"Async function {func_name} failed on attempt {attempt + 1}/{max_attempts}. "
                            f"Retrying in {current_delay:.2f}s. Error: {e}"
                        )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but just in case
            raise last_exception if last_exception else RuntimeError("Async retry decorator failed")
        
        return wrapper
    
    return decorator


def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Safely load JSON data.
    
    Args:
        data: JSON string to load
        default: Default value if loading fails
        
    Returns:
        Loaded JSON data or default value
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, indent: Optional[int] = None) -> str:
    """
    Safely dump object to JSON string.
    
    Args:
        obj: Object to dump
        indent: JSON indentation
        
    Returns:
        JSON string or empty string on failure
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count to human-readable string.
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 30m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_nested_value(data: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """
    Get nested value from dictionary using dot notation.
    
    Args:
        data: Dictionary to search
        keys: Dot-separated keys (e.g., "user.profile.name")
        default: Default value if key not found
        
    Returns:
        Nested value or default
    """
    key_list = keys.split('.')
    current = data
    
    try:
        for key in key_list:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except (KeyError, TypeError):
        return default


def set_nested_value(data: Dict[str, Any], keys: str, value: Any) -> None:
    """
    Set nested value in dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        keys: Dot-separated keys (e.g., "user.profile.name")
        value: Value to set
    """
    key_list = keys.split('.')
    current = data
    
    for key in key_list[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[key_list[-1]] = value


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_path(path: Union[str, Path], must_exist: bool = True, create_dirs: bool = False) -> Path:
    """
    Validate and optionally create directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        create_dirs: Whether to create directories if they don't exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path validation fails
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path_obj}")
    
    if create_dirs and not path_obj.exists():
        # If path has a file extension, create parent directory
        # If path doesn't have an extension, create it as a directory
        if path_obj.suffix:
            # Has file extension, create parent directory
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        else:
            # No file extension, create as directory
            path_obj.mkdir(parents=True, exist_ok=True)
    
    return path_obj


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary containing system information
    """
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'free': psutil.disk_usage('/').free,
        },
    }


def calculate_hash(data: Union[str, bytes]) -> str:
    """
    Calculate SHA256 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    import hashlib
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


class RateLimiter:
    """
    Simple rate limiter implementation.
    """
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        
    def is_allowed(self) -> bool:
        """
        Check if a call is allowed.
        
        Returns:
            True if call is allowed, False otherwise
        """
        current_time = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if current_time - call_time < self.time_window]
        
        # Check if we can make a new call
        if len(self.calls) < self.max_calls:
            self.calls.append(current_time)
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self.calls.clear()
    
    def get_wait_time(self) -> float:
        """
        Get time to wait until next call is allowed.
        
        Returns:
            Time to wait in seconds
        """
        if len(self.calls) < self.max_calls:
            return 0.0
        
        current_time = time.time()
        oldest_call = min(self.calls)
        return max(0.0, self.time_window - (current_time - oldest_call))