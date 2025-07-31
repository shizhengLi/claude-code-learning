"""
Error handling utilities for the context manager system.
"""

import traceback
from typing import Optional, Dict, Any, Type
from .logging import get_logger


logger = get_logger(__name__)


class ContextManagerError(Exception):
    """Base exception for context manager errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize context manager error.
        
        Args:
            message: Error message
            error_code: Optional error code
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = None
        
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            base_msg += f" (Context: {self.context})"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc(),
        }


class MemoryError(ContextManagerError):
    """Exception raised for memory-related errors."""
    
    def __init__(self, message: str, memory_type: Optional[str] = None, **kwargs):
        """
        Initialize memory error.
        
        Args:
            message: Error message
            memory_type: Type of memory (short_term, medium_term, long_term)
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.memory_type = memory_type
        
        if memory_type:
            self.context['memory_type'] = memory_type


class ToolError(ContextManagerError):
    """Exception raised for tool-related errors."""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, 
                 execution_id: Optional[str] = None, **kwargs):
        """
        Initialize tool error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that failed
            execution_id: Execution ID
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.execution_id = execution_id
        
        if tool_name:
            self.context['tool_name'] = tool_name
        if execution_id:
            self.context['execution_id'] = execution_id


class ConfigurationError(ContextManagerError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.config_key = config_key
        
        if config_key:
            self.context['config_key'] = config_key


class StorageError(ContextManagerError):
    """Exception raised for storage-related errors."""
    
    def __init__(self, message: str, storage_path: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        """
        Initialize storage error.
        
        Args:
            message: Error message
            storage_path: Path of the storage that failed
            operation: Operation that failed (read, write, delete, etc.)
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.storage_path = storage_path
        self.operation = operation
        
        if storage_path:
            self.context['storage_path'] = storage_path
        if operation:
            self.context['operation'] = operation


class CompressionError(ContextManagerError):
    """Exception raised for compression-related errors."""
    
    def __init__(self, message: str, compression_type: Optional[str] = None, **kwargs):
        """
        Initialize compression error.
        
        Args:
            message: Error message
            compression_type: Type of compression that failed
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.compression_type = compression_type
        
        if compression_type:
            self.context['compression_type'] = compression_type


class TimeoutError(ContextManagerError):
    """Exception raised for timeout-related errors."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            timeout_duration: Timeout duration in seconds
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        
        if timeout_duration:
            self.context['timeout_duration'] = timeout_duration


class ValidationError(ContextManagerError):
    """Exception raised for validation-related errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Optional[Any] = None, **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        
        if field_name:
            self.context['field_name'] = field_name
        if field_value is not None:
            self.context['field_value'] = str(field_value)


class RateLimitError(ContextManagerError):
    """Exception raised for rate limit-related errors."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Time to wait before retrying (seconds)
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        
        if retry_after:
            self.context['retry_after'] = retry_after


def handle_error(
    error: Exception,
    reraise: bool = False,
    error_types: Optional[Dict[Type[Exception], str]] = None,
    default_message: str = "An unexpected error occurred"
) -> Optional[ContextManagerError]:
    """
    Handle and possibly convert errors.
    
    Args:
        error: The original error
        reraise: Whether to reraise the error after handling
        error_types: Mapping of error types to error codes
        default_message: Default error message for unknown errors
        
    Returns:
        ContextManagerError if conversion was successful, None otherwise
        
    Raises:
        The original error if reraise is True
    """
    error_types = error_types or {}
    
    try:
        # If it's already a ContextManagerError, just log it
        if isinstance(error, ContextManagerError):
            logger.error(f"Context manager error: {error}", extra={'context': error.context})
            if reraise:
                raise error
            return error
        
        # Convert known error types
        for error_type, error_code in error_types.items():
            if isinstance(error, error_type):
                cm_error = ContextManagerError(
                    message=str(error),
                    error_code=error_code,
                    context={'original_error': type(error).__name__}
                )
                logger.error(f"Converted error: {cm_error}", extra={'context': cm_error.context})
                if reraise:
                    raise cm_error
                return cm_error
        
        # Handle unknown errors
        cm_error = ContextManagerError(
            message=default_message,
            error_code="UNKNOWN_ERROR",
            context={
                'original_error': type(error).__name__,
                'original_message': str(error)
            }
        )
        logger.error(f"Unknown error: {cm_error}", extra={'context': cm_error.context})
        
        if reraise:
            raise cm_error
        
        return cm_error
        
    except Exception as handling_error:
        logger.error(f"Error while handling error: {handling_error}")
        if reraise:
            raise error
        return None


def log_error(
    error: Exception,
    logger_instance = None,
    level: str = "ERROR",
    include_traceback: bool = True
) -> None:
    """
    Log an error with appropriate context.
    
    Args:
        error: The error to log
        logger_instance: Logger instance to use (defaults to module logger)
        level: Log level
        include_traceback: Whether to include traceback
    """
    if logger_instance is None:
        logger_instance = logger
    
    log_method = getattr(logger_instance, level.lower(), logger_instance.error)
    
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error)
    }
    
    if isinstance(error, ContextManagerError):
        error_info.update({
            'error_code': error.error_code,
            'context': error.context
        })
    
    message = f"Error occurred: {error_info['error_type']}: {error_info['error_message']}"
    
    if include_traceback:
        error_info['traceback'] = traceback.format_exc()
    
    log_method(message, extra={'error_info': error_info})


def create_error_response(
    error: Exception,
    include_context: bool = True,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    
    Args:
        error: The error to convert
        include_context: Whether to include error context
        include_traceback: Whether to include traceback
        
    Returns:
        Dictionary containing error information
    """
    response = {
        'success': False,
        'error': {
            'type': type(error).__name__,
            'message': str(error)
        }
    }
    
    if isinstance(error, ContextManagerError):
        response['error']['code'] = error.error_code
        if include_context and error.context:
            response['error']['context'] = error.context
    
    if include_traceback:
        response['error']['traceback'] = traceback.format_exc()
    
    return response


class ErrorHandler:
    """
    Centralized error handler for the context manager system.
    """
    
    def __init__(self, logger_instance=None):
        """
        Initialize error handler.
        
        Args:
            logger_instance: Logger instance to use
        """
        self.logger = logger_instance or logger
        self.error_counts = {}
        self.error_callbacks = {}
        
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error and return a standardized response.
        
        Args:
            error: The error to handle
            context: Additional context information
            
        Returns:
            Error response dictionary
        """
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create response
        response = create_error_response(error, include_context=True)
        
        # Add additional context
        if context:
            if 'context' not in response['error']:
                response['error']['context'] = {}
            response['error']['context'].update(context)
        
        # Log the error
        log_error(error, self.logger, include_traceback=True)
        
        # Call error callbacks
        self._call_error_callbacks(error, response)
        
        return response
    
    def add_error_callback(self, error_type: Type[Exception], callback) -> None:
        """
        Add a callback for specific error types.
        
        Args:
            error_type: Type of error to handle
            callback: Callback function to call
        """
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        self.error_callbacks[error_type].append(callback)
    
    def _call_error_callbacks(self, error: Exception, response: Dict[str, Any]) -> None:
        """
        Call error callbacks for the given error.
        
        Args:
            error: The error that occurred
            response: The error response
        """
        error_type = type(error)
        called_callbacks = set()
        
        # Call callbacks for this specific error type
        for callback in self.error_callbacks.get(error_type, []):
            try:
                callback(error, response)
                called_callbacks.add(callback)
            except Exception as callback_error:
                self.logger.error(f"Error in error callback: {callback_error}")
        
        # Call callbacks for base ContextManagerError (if not already called)
        if isinstance(error, ContextManagerError) and error_type != ContextManagerError:
            for callback in self.error_callbacks.get(ContextManagerError, []):
                if callback not in called_callbacks:
                    try:
                        callback(error, response)
                    except Exception as callback_error:
                        self.logger.error(f"Error in error callback: {callback_error}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts,
            'unique_error_types': len(self.error_counts)
        }
    
    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()


# Global error handler instance
global_error_handler = ErrorHandler()