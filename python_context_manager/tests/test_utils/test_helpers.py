"""
Tests for helper utilities.
"""

import pytest
import time
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from context_manager.utils.helpers import (
    ContextManagerError, MemoryError, ToolError, ConfigurationError,
    StorageError, CompressionError, Timer, retry, async_retry,
    safe_json_loads, safe_json_dumps, format_bytes, format_duration,
    truncate_string, get_nested_value, set_nested_value, merge_dicts,
    validate_path, get_system_info, calculate_hash, RateLimiter
)


class TestTimer:
    """Test cases for Timer class."""
    
    def test_timer_creation(self):
        """Test basic timer creation."""
        timer = Timer("test_operation")
        
        assert timer.name == "test_operation"
        assert timer.start_time is None
        assert timer.end_time is None
        assert timer.elapsed == 0.0
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        timer = Timer("test_operation")
        
        with timer:
            # Simulate some work
            time.sleep(0.01)
            assert timer.start_time is not None
            assert timer.end_time is None
        
        # After context exit
        assert timer.end_time is not None
        assert timer.elapsed > 0
        assert timer.elapsed < 1.0  # Should be very short
    
    def test_timer_string_representation(self):
        """Test timer string representation."""
        timer = Timer("test_operation")
        
        # Before timing
        assert "test_operation: 0.0000s" in str(timer)
        
        # After timing
        with timer:
            time.sleep(0.01)
        
        timer_str = str(timer)
        assert "test_operation:" in timer_str
        assert "s" in timer_str
    
    def test_timer_with_custom_logger(self):
        """Test timer with custom logger."""
        custom_logger = Mock()
        timer = Timer("test_operation", custom_logger)
        
        with timer:
            time.sleep(0.01)
        
        custom_logger.info.assert_called_once()
        assert "test_operation took" in custom_logger.info.call_args[0][0]


class TestRetryDecorator:
    """Test cases for retry decorator."""
    
    def test_retry_successful_execution(self):
        """Test retry with successful execution."""
        mock_func = Mock(return_value="success")
        
        decorated_func = retry(max_attempts=3)(mock_func)
        
        result = decorated_func("arg1", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
    
    def test_retry_failure_then_success(self):
        """Test retry with failure then success."""
        mock_func = Mock(side_effect=[ValueError("Error"), "success"])
        
        decorated_func = retry(max_attempts=3)(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_all_attempts_fail(self):
        """Test retry when all attempts fail."""
        mock_func = Mock(side_effect=ValueError("Persistent error"))
        
        decorated_func = retry(max_attempts=3)(mock_func)
        
        with pytest.raises(ValueError, match="Persistent error"):
            decorated_func()
        
        assert mock_func.call_count == 3
    
    def test_retry_with_custom_delay_and_backoff(self):
        """Test retry with custom delay and backoff."""
        mock_func = Mock(side_effect=[ValueError("Error"), ValueError("Error"), "success"])
        
        decorated_func = retry(
            max_attempts=3,
            delay=0.1,
            backoff=2.0
        )(mock_func)
        
        start_time = time.time()
        result = decorated_func()
        end_time = time.time()
        
        assert result == "success"
        assert mock_func.call_count == 3
        # Should have taken at least 0.1 + 0.2 = 0.3 seconds
        assert end_time - start_time >= 0.25
    
    def test_retry_with_custom_exceptions(self):
        """Test retry with custom exception types."""
        mock_func = Mock(side_effect=[ValueError("Error"), "success"])
        
        decorated_func = retry(
            max_attempts=3,
            exceptions=(ValueError,)
        )(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_with_non_matching_exception(self):
        """Test retry with non-matching exception type."""
        mock_func = Mock(side_effect=TypeError("Type error"))
        
        decorated_func = retry(
            max_attempts=3,
            exceptions=(ValueError,)
        )(mock_func)
        
        with pytest.raises(TypeError, match="Type error"):
            decorated_func()
        
        assert mock_func.call_count == 1  # Should not retry
    
    def test_retry_with_logger(self):
        """Test retry with logger."""
        mock_func = Mock(side_effect=[ValueError("Error"), "success"])
        mock_logger = Mock()
        
        decorated_func = retry(max_attempts=3, logger=mock_logger)(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 2
        mock_logger.warning.assert_called_once()
    
    def test_retry_function_name_in_logs(self):
        """Test that function name appears in retry logs."""
        def failing_function():
            raise ValueError("Test error")
        
        mock_logger = Mock()
        decorated_func = retry(max_attempts=2, logger=mock_logger)(failing_function)
        
        with pytest.raises(ValueError):
            decorated_func()
        
        # Check that function name is in log messages
        log_calls = mock_logger.warning.call_args_list + mock_logger.error.call_args_list
        assert any("failing_function" in str(call) for call in log_calls)


class TestAsyncRetryDecorator:
    """Test cases for async retry decorator."""
    
    @pytest.mark.asyncio
    async def test_async_retry_successful_execution(self):
        """Test async retry with successful execution."""
        async def async_func():
            return "async_success"
        
        decorated_func = async_retry(max_attempts=3)(async_func)
        
        result = await decorated_func()
        
        assert result == "async_success"
    
    @pytest.mark.asyncio
    async def test_async_retry_failure_then_success(self):
        """Test async retry with failure then success."""
        call_count = 0
        
        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Async error")
            return "async_success"
        
        decorated_func = async_retry(max_attempts=3)(async_func)
        
        result = await decorated_func()
        
        assert result == "async_success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_retry_all_attempts_fail(self):
        """Test async retry when all attempts fail."""
        async def async_func():
            raise ValueError("Persistent async error")
        
        decorated_func = async_retry(max_attempts=3)(async_func)
        
        with pytest.raises(ValueError, match="Persistent async error"):
            await decorated_func()
    
    @pytest.mark.asyncio
    async def test_async_retry_with_delay(self):
        """Test async retry with custom delay."""
        call_count = 0
        
        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Async error")
            return "async_success"
        
        decorated_func = async_retry(
            max_attempts=3,
            delay=0.1
        )(async_func)
        
        start_time = time.time()
        result = await decorated_func()
        end_time = time.time()
        
        assert result == "async_success"
        assert call_count == 3
        assert end_time - start_time >= 0.15  # At least 2 delays


class TestJsonUtilities:
    """Test cases for JSON utility functions."""
    
    def test_safe_json_loads_valid_json(self):
        """Test safe JSON loads with valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_loads(json_str)
        
        assert result == {"key": "value", "number": 42}
    
    def test_safe_json_loads_invalid_json(self):
        """Test safe JSON loads with invalid JSON."""
        invalid_json = '{"invalid": json}'
        result = safe_json_loads(invalid_json, default={"fallback": True})
        
        assert result == {"fallback": True}
    
    def test_safe_json_loads_none_default(self):
        """Test safe JSON loads with None default."""
        invalid_json = '{"invalid": json}'
        result = safe_json_loads(invalid_json)
        
        assert result is None
    
    def test_safe_json_dumps_valid_object(self):
        """Test safe JSON dumps with valid object."""
        obj = {"key": "value", "number": 42}
        result = safe_json_dumps(obj, indent=2)
        
        assert '"key": "value"' in result
        assert '"number": 42' in result
    
    def test_safe_json_dumps_invalid_object(self):
        """Test safe JSON dumps with invalid object."""
        # Create an object that can't be serialized
        class UnserializableObject:
            pass
        
        obj = {"key": UnserializableObject()}
        result = safe_json_dumps(obj)
        
        assert result == ""  # Should return empty string on failure
    
    def test_safe_json_dumps_without_indent(self):
        """Test safe JSON dumps without indentation."""
        obj = {"key": "value"}
        result = safe_json_dumps(obj)
        
        assert result == '{"key": "value"}'


class TestFormattingUtilities:
    """Test cases for formatting utility functions."""
    
    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(500) == "500.0 B"
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(1048576) == "1.0 MB"
        assert format_bytes(1073741824) == "1.0 GB"
        assert format_bytes(1099511627776) == "1.0 TB"
        assert format_bytes(1125899906842624) == "1.0 PB"
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3661) == "1h 1m 1s"
        assert format_duration(7320) == "2h 2m 0s"
    
    def test_truncate_string(self):
        """Test string truncation."""
        text = "This is a long string that needs to be truncated"
        
        # No truncation needed
        result = truncate_string(text, 100)
        assert result == text
        
        # Truncation needed
        result = truncate_string(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
        
        # Custom suffix
        result = truncate_string(text, 20, suffix "***")
        assert len(result) == 20
        assert result.endswith("***")


class TestDictionaryUtilities:
    """Test cases for dictionary utility functions."""
    
    def test_get_nested_value(self):
        """Test getting nested value."""
        data = {
            "user": {
                "profile": {
                    "name": "Alice",
                    "age": 30
                }
            }
        }
        
        assert get_nested_value(data, "user.profile.name") == "Alice"
        assert get_nested_value(data, "user.profile.age") == 30
        assert get_nested_value(data, "user.profile.nonexistent") is None
        assert get_nested_value(data, "user.profile.nonexistent", "default") == "default"
    
    def test_get_nested_value_invalid_path(self):
        """Test getting nested value with invalid path."""
        data = {"simple": "value"}
        
        assert get_nested_value(data, "nonexistent.key") is None
        assert get_nested_value(data, "simple.nonexistent") is None
    
    def test_set_nested_value(self):
        """Test setting nested value."""
        data = {}
        
        set_nested_value(data, "user.profile.name", "Alice")
        assert data["user"]["profile"]["name"] == "Alice"
        
        set_nested_value(data, "user.profile.age", 30)
        assert data["user"]["profile"]["age"] == 30
    
    def test_set_nested_value_existing_path(self):
        """Test setting nested value with existing path."""
        data = {"user": {"profile": {"name": "Alice"}}}
        
        set_nested_value(data, "user.profile.name", "Bob")
        assert data["user"]["profile"]["name"] == "Bob"
    
    def test_merge_dicts(self):
        """Test dictionary merging."""
        dict1 = {"a": 1, "b": {"x": 10, "y": 20}}
        dict2 = {"b": {"y": 25, "z": 30}, "c": 3}
        
        result = merge_dicts(dict1, dict2)
        
        assert result["a"] == 1
        assert result["b"]["x"] == 10  # From dict1
        assert result["b"]["y"] == 25  # From dict2 (overwritten)
        assert result["b"]["z"] == 30  # From dict2
        assert result["c"] == 3  # From dict2
    
    def test_merge_dicts_no_conflict(self):
        """Test dictionary merging without conflicts."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        
        result = merge_dicts(dict1, dict2)
        
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}


class TestPathUtilities:
    """Test cases for path utility functions."""
    
    def test_validate_path_existing(self):
        """Test validating existing path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            
            result = validate_path(path, must_exist=True)
            
            assert result == path
    
    def test_validate_path_non_existing(self):
        """Test validating non-existing path."""
        non_existent = "/tmp/non_existent_path_12345"
        
        with pytest.raises(ValueError, match="Path does not exist"):
            validate_path(non_existent, must_exist=True)
    
    def test_validate_path_create_dirs(self):
        """Test validating path with directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new" / "subdir"
            
            # Directory should not exist initially
            assert not new_dir.exists()
            
            result = validate_path(new_dir, must_exist=False, create_dirs=True)
            
            # Directory should be created
            assert new_dir.exists()
            assert result == new_dir
    
    def test_validate_path_string_input(self):
        """Test validating path with string input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_str = temp_dir
            
            result = validate_path(path_str, must_exist=True)
            
            assert isinstance(result, Path)
            assert result == Path(temp_dir)


class TestSystemUtilities:
    """Test cases for system utility functions."""
    
    @patch('context_manager.utils.helpers.psutil')
    @patch('context_manager.utils.helpers.platform')
    def test_get_system_info(self, mock_platform, mock_psutil):
        """Test getting system information."""
        # Mock platform functions
        mock_platform.platform.return_value = "Linux-5.4.0"
        mock_platform.python_version.return_value = "3.9.0"
        
        # Mock psutil functions
        mock_psutil.cpu_count.return_value = 8
        mock_memory = Mock()
        mock_memory.total = 16777216  # 16GB
        mock_memory.available = 8388608  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 1000000000
        mock_disk.used = 500000000
        mock_disk.free = 500000000
        mock_psutil.disk_usage.return_value = mock_disk
        
        result = get_system_info()
        
        assert result['platform'] == "Linux-5.4.0"
        assert result['python_version'] == "3.9.0"
        assert result['cpu_count'] == 8
        assert result['memory_total'] == 16777216
        assert result['memory_available'] == 8388608
        assert result['disk_usage']['total'] == 1000000000
        assert result['disk_usage']['used'] == 500000000
        assert result['disk_usage']['free'] == 500000000
    
    def test_calculate_hash_string(self):
        """Test calculating hash of string."""
        text = "Hello, world!"
        result = calculate_hash(text)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hash length
        assert result == "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
    
    def test_calculate_hash_bytes(self):
        """Test calculating hash of bytes."""
        data = b"Hello, world!"
        result = calculate_hash(data)
        
        assert isinstance(result, str)
        assert len(result) == 64
        assert result == "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"


class TestRateLimiter:
    """Test cases for RateLimiter class."""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter creation."""
        limiter = RateLimiter(max_calls=5, time_window=60)
        
        assert limiter.max_calls == 5
        assert limiter.time_window == 60
        assert limiter.calls == []
    
    def test_rate_limiter_allowed_calls(self):
        """Test rate limiter allows calls within limit."""
        limiter = RateLimiter(max_calls=3, time_window=60)
        
        # Should allow first 3 calls
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        
        # Should deny 4th call
        assert limiter.is_allowed() is False
    
    def test_rate_limiter_time_window_reset(self):
        """Test rate limiter resets after time window."""
        limiter = RateLimiter(max_calls=2, time_window=0.1)  # 100ms window
        
        # Use up calls
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
        
        # Wait for time window to pass
        time.sleep(0.15)
        
        # Should allow calls again
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
    
    def test_rate_limiter_get_wait_time(self):
        """Test rate limiter wait time calculation."""
        limiter = RateLimiter(max_calls=2, time_window=0.1)
        
        # No wait time initially
        assert limiter.get_wait_time() == 0.0
        
        # Use up calls
        limiter.is_allowed()
        limiter.is_allowed()
        
        # Should have wait time
        wait_time = limiter.get_wait_time()
        assert wait_time > 0.0
        assert wait_time <= 0.1
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset."""
        limiter = RateLimiter(max_calls=2, time_window=60)
        
        # Use up calls
        limiter.is_allowed()
        limiter.is_allowed()
        assert limiter.is_allowed() is False
        
        # Reset limiter
        limiter.reset()
        
        # Should allow calls again
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
    
    def test_rate_limiter_cleanup_old_calls(self):
        """Test rate limiter cleans up old calls."""
        limiter = RateLimiter(max_calls=2, time_window=0.1)
        
        # Make some calls
        limiter.is_allowed()
        limiter.is_allowed()
        assert len(limiter.calls) == 2
        
        # Wait for calls to expire
        time.sleep(0.15)
        
        # Should clean up old calls
        limiter.is_allowed()
        assert len(limiter.calls) == 1  # Only the new call


# Import tempfile for tests
import tempfile