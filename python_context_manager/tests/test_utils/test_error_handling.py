"""
Tests for error handling utilities.
"""

import pytest
import time
from unittest.mock import Mock, patch
from context_manager.utils.error_handling import (
    ContextManagerError, MemoryError, ToolError, ConfigurationError,
    StorageError, CompressionError, TimeoutError, ValidationError,
    RateLimitError, handle_error, log_error, create_error_response,
    ErrorHandler, global_error_handler
)


class TestContextManagerError:
    """Test cases for ContextManagerError class."""
    
    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = ContextManagerError("Test error message")
        
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}
        assert error.timestamp is None
    
    def test_error_with_code_and_context(self):
        """Test error with error code and context."""
        context = {"operation": "test", "user_id": "123"}
        error = ContextManagerError(
            "Test error",
            error_code="TEST_ERROR",
            context=context
        )
        
        assert error.error_code == "TEST_ERROR"
        assert error.context == context
    
    def test_error_string_representation(self):
        """Test error string representation."""
        # Basic error
        error = ContextManagerError("Basic error")
        assert str(error) == "Basic error"
        
        # Error with code
        error_with_code = ContextManagerError("Error with code", error_code="ERR_001")
        assert str(error_with_code) == "[ERR_001] Error with code"
        
        # Error with context
        error_with_context = ContextManagerError(
            "Error with context",
            context={"key": "value"}
        )
        assert "Error with context" in str(error_with_context)
        assert "Context:" in str(error_with_context)
    
    def test_error_to_dict(self):
        """Test error dictionary conversion."""
        error = ContextManagerError(
            "Test error",
            error_code="TEST_001",
            context={"operation": "test"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['error_type'] == 'ContextManagerError'
        assert error_dict['message'] == 'Test error'
        assert error_dict['error_code'] == 'TEST_001'
        assert error_dict['context'] == {'operation': 'test'}
        assert 'timestamp' in error_dict
        assert 'traceback' in error_dict


class TestSpecializedErrors:
    """Test cases for specialized error classes."""
    
    def test_memory_error(self):
        """Test MemoryError creation."""
        error = MemoryError("Memory full", memory_type="short_term")
        
        assert error.memory_type == "short_term"
        assert error.context['memory_type'] == "short_term"
        assert isinstance(error, ContextManagerError)
    
    def test_tool_error(self):
        """Test ToolError creation."""
        error = ToolError(
            "Tool failed",
            tool_name="test_tool",
            execution_id="exec_123"
        )
        
        assert error.tool_name == "test_tool"
        assert error.execution_id == "exec_123"
        assert error.context['tool_name'] == "test_tool"
        assert error.context['execution_id'] == "exec_123"
    
    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError("Invalid config", config_key="max_tokens")
        
        assert error.config_key == "max_tokens"
        assert error.context['config_key'] == "max_tokens"
    
    def test_storage_error(self):
        """Test StorageError creation."""
        error = StorageError(
            "Storage failed",
            storage_path="/tmp/test.json",
            operation="write"
        )
        
        assert error.storage_path == "/tmp/test.json"
        assert error.operation == "write"
        assert error.context['storage_path'] == "/tmp/test.json"
        assert error.context['operation'] == "write"
    
    def test_compression_error(self):
        """Test CompressionError creation."""
        error = CompressionError("Compression failed", compression_type="gzip")
        
        assert error.compression_type == "gzip"
        assert error.context['compression_type'] == "gzip"
    
    def test_timeout_error(self):
        """Test TimeoutError creation."""
        error = TimeoutError("Operation timed out", timeout_duration=30.0)
        
        assert error.timeout_duration == 30.0
        assert error.context['timeout_duration'] == 30.0
    
    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError(
            "Invalid field",
            field_name="email",
            field_value="invalid_email"
        )
        
        assert error.field_name == "email"
        assert error.field_value == "invalid_email"
        assert error.context['field_name'] == "email"
        assert error.context['field_value'] == "invalid_email"
    
    def test_rate_limit_error(self):
        """Test RateLimitError creation."""
        error = RateLimitError("Rate limit exceeded", retry_after=5.0)
        
        assert error.retry_after == 5.0
        assert error.context['retry_after'] == 5.0


class TestErrorHandlingFunctions:
    """Test cases for error handling functions."""
    
    def test_handle_context_manager_error(self):
        """Test handling ContextManagerError."""
        error = ContextManagerError("Test error", error_code="TEST_001")
        
        with patch('context_manager.utils.error_handling.logger') as mock_logger:
            result = handle_error(error, reraise=False)
            
            assert result == error
            mock_logger.error.assert_called_once()
    
    def test_handle_known_error_type(self):
        """Test handling known error type with conversion."""
        original_error = ValueError("Invalid value")
        error_types = {ValueError: "VALIDATION_ERROR"}
        
        with patch('context_manager.utils.error_handling.logger') as mock_logger:
            result = handle_error(
                original_error,
                reraise=False,
                error_types=error_types
            )
            
            assert isinstance(result, ContextManagerError)
            assert result.error_code == "VALIDATION_ERROR"
            assert result.context['original_error'] == 'ValueError'
            mock_logger.error.assert_called_once()
    
    def test_handle_unknown_error(self):
        """Test handling unknown error type."""
        original_error = RuntimeError("Unknown error")
        
        with patch('context_manager.utils.error_handling.logger') as mock_logger:
            result = handle_error(original_error, reraise=False)
            
            assert isinstance(result, ContextManagerError)
            assert result.error_code == "UNKNOWN_ERROR"
            assert result.context['original_error'] == 'RuntimeError'
            mock_logger.error.assert_called_once()
    
    def test_handle_error_with_reraise(self):
        """Test handling error with reraise."""
        error = ContextManagerError("Test error")
        
        with pytest.raises(ContextManagerError):
            handle_error(error, reraise=True)
    
    def test_log_error(self):
        """Test error logging."""
        error = ContextManagerError("Test error", error_code="TEST_001")
        
        with patch('context_manager.utils.error_handling.logger') as mock_logger:
            log_error(error)
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0]
            assert "Test error" in call_args[0]
    
    def test_log_error_with_custom_logger(self):
        """Test error logging with custom logger."""
        error = ContextManagerError("Test error")
        custom_logger = Mock()
        
        log_error(error, custom_logger)
        
        custom_logger.error.assert_called_once()
    
    def test_create_error_response(self):
        """Test creating error response."""
        error = ContextManagerError(
            "Test error",
            error_code="TEST_001",
            context={"operation": "test"}
        )
        
        response = create_error_response(error)
        
        assert response['success'] is False
        assert response['error']['type'] == 'ContextManagerError'
        assert response['error']['message'] == '[TEST_001] Test error (Context: {\'operation\': \'test\'})'
        assert response['error']['code'] == 'TEST_001'
        assert response['error']['context'] == {'operation': 'test'}
    
    def test_create_error_response_with_traceback(self):
        """Test creating error response with traceback."""
        error = ContextManagerError("Test error")
        
        response = create_error_response(error, include_traceback=True)
        
        assert 'traceback' in response['error']
    
    def test_create_error_response_for_standard_exception(self):
        """Test creating error response for standard exception."""
        error = ValueError("Standard error")
        
        response = create_error_response(error)
        
        assert response['success'] is False
        assert response['error']['type'] == 'ValueError'
        assert response['error']['message'] == 'Standard error'
        assert 'code' not in response['error']


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def test_error_handler_creation(self):
        """Test basic error handler creation."""
        handler = ErrorHandler()
        
        assert handler.error_counts == {}
        assert handler.error_callbacks == {}
    
    def test_error_handler_with_custom_logger(self):
        """Test error handler with custom logger."""
        custom_logger = Mock()
        handler = ErrorHandler(custom_logger)
        
        assert handler.logger == custom_logger
    
    def test_handle_error(self):
        """Test error handling through ErrorHandler."""
        handler = ErrorHandler()
        error = ContextManagerError("Test error", error_code="TEST_001")
        
        with patch.object(handler, 'logger') as mock_logger:
            response = handler.handle_error(error)
            
            assert response['success'] is False
            assert response['error']['message'] == '[TEST_001] Test error'
            assert handler.error_counts['ContextManagerError'] == 1
            mock_logger.error.assert_called_once()
    
    def test_handle_error_with_context(self):
        """Test error handling with additional context."""
        handler = ErrorHandler()
        error = ContextManagerError("Test error")
        
        response = handler.handle_error(error, context={"user_id": "123"})
        
        assert response['error']['context']['user_id'] == "123"
    
    def test_add_error_callback(self):
        """Test adding error callbacks."""
        handler = ErrorHandler()
        callback = Mock()
        
        handler.add_error_callback(ValueError, callback)
        
        assert ValueError in handler.error_callbacks
        assert callback in handler.error_callbacks[ValueError]
    
    def test_error_callback_execution(self):
        """Test error callback execution."""
        handler = ErrorHandler()
        callback = Mock()
        
        handler.add_error_callback(ValueError, callback)
        
        error = ValueError("Test error")
        handler.handle_error(error)
        
        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == error
        assert 'error' in call_args[1]
    
    def test_context_manager_error_callback(self):
        """Test callback execution for ContextManagerError."""
        handler = ErrorHandler()
        callback = Mock()
        
        handler.add_error_callback(ContextManagerError, callback)
        
        error = ContextManagerError("Test error")
        handler.handle_error(error)
        
        callback.assert_called_once()
    
    def test_get_error_stats(self):
        """Test getting error statistics."""
        handler = ErrorHandler()
        
        # Handle some errors
        handler.handle_error(ValueError("Error 1"))
        handler.handle_error(ValueError("Error 2"))
        handler.handle_error(RuntimeError("Error 3"))
        
        stats = handler.get_error_stats()
        
        assert stats['total_errors'] == 3
        assert stats['error_counts']['ValueError'] == 2
        assert stats['error_counts']['RuntimeError'] == 1
        assert stats['unique_error_types'] == 2
    
    def test_reset_stats(self):
        """Test resetting error statistics."""
        handler = ErrorHandler()
        
        handler.handle_error(ValueError("Test error"))
        assert handler.error_counts['ValueError'] == 1
        
        handler.reset_stats()
        assert handler.error_counts == {}
    
    def test_callback_error_handling(self):
        """Test handling of errors in callbacks."""
        handler = ErrorHandler()
        
        def failing_callback(error, response):
            raise RuntimeError("Callback failed")
        
        handler.add_error_callback(ValueError, failing_callback)
        
        # Should not raise exception
        response = handler.handle_error(ValueError("Test error"))
        assert response['success'] is False


class TestGlobalErrorHandler:
    """Test cases for global error handler."""
    
    def test_global_error_handler_exists(self):
        """Test that global error handler exists."""
        assert global_error_handler is not None
        assert isinstance(global_error_handler, ErrorHandler)
    
    def test_global_error_handler_usage(self):
        """Test using global error handler."""
        error = ContextManagerError("Test error")
        
        initial_count = global_error_handler.get_error_stats()['total_errors']
        response = global_error_handler.handle_error(error)
        
        assert response['success'] is False
        assert global_error_handler.get_error_stats()['total_errors'] == initial_count + 1