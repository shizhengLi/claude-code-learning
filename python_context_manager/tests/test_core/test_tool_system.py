"""
Tests for the tool system.

This module tests the comprehensive tool system including:
- Tool registration and discovery
- Tool execution with various strategies
- Tool performance monitoring
- Tool security and validation
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from unittest.mock import Mock, patch, AsyncMock
from context_manager.core.tool_system import (
    ToolInterface, ToolRegistry, ToolExecutor, ToolMetadata,
    ToolCategory, ToolPermission, ToolExecutionContext,
    FunctionTool, tool, RateLimiter
)
from context_manager.core.tool_manager import (
    ToolManager, ToolExecutionRequest, ToolExecutionResponse
)
from context_manager.core.models import ToolStatus, ToolResult
from context_manager.core.config import ContextManagerConfig


class TestToolMetadata:
    """Test cases for ToolMetadata."""
    
    def test_tool_metadata_creation(self):
        """Test basic tool metadata creation."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.UTILITY
        )
        
        assert metadata.name == "test_tool"
        assert metadata.description == "Test tool"
        assert metadata.category == ToolCategory.UTILITY
        assert metadata.version == "1.0.0"
        assert metadata.is_active is True
        assert metadata.execution_count == 0
        assert metadata.success_count == 0
        assert metadata.failure_count == 0
    
    def test_tool_metadata_with_custom_values(self):
        """Test tool metadata with custom values."""
        metadata = ToolMetadata(
            name="custom_tool",
            description="Custom tool",
            category=ToolCategory.DATA_PROCESSING,
            version="2.0.0",
            author="Test Author",
            tags=["test", "custom"],
            permission=ToolPermission.RESTRICTED,
            timeout=60.0,
            max_retries=5,
            rate_limit=10,
            rate_window=30.0,
            dependencies=["tool1", "tool2"]
        )
        
        assert metadata.name == "custom_tool"
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "custom"]
        assert metadata.permission == ToolPermission.RESTRICTED
        assert metadata.timeout == 60.0
        assert metadata.max_retries == 5
        assert metadata.rate_limit == 10
        assert metadata.rate_window == 30.0
        assert metadata.dependencies == ["tool1", "tool2"]


class TestToolExecutionContext:
    """Test cases for ToolExecutionContext."""
    
    def test_context_creation(self):
        """Test basic context creation."""
        context = ToolExecutionContext(
            execution_id="test_exec_123",
            tool_name="test_tool",
            parameters={"param1": "value1"}
        )
        
        assert context.execution_id == "test_exec_123"
        assert context.tool_name == "test_tool"
        assert context.parameters == {"param1": "value1"}
        assert context.user_id is None
        assert context.session_id is None
        assert isinstance(context.start_time, datetime)
    
    def test_context_with_all_fields(self):
        """Test context with all fields."""
        context = ToolExecutionContext(
            execution_id="exec_456",
            tool_name="complex_tool",
            parameters={"param1": "value1", "param2": 42},
            user_id="user_123",
            session_id="session_456",
            metadata={"custom": "data"}
        )
        
        assert context.execution_id == "exec_456"
        assert context.tool_name == "complex_tool"
        assert context.parameters == {"param1": "value1", "param2": 42}
        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.metadata == {"custom": "data"}


class MockTool(ToolInterface):
    """Mock tool for testing."""
    
    def __init__(self, metadata: ToolMetadata, should_fail: bool = False):
        super().__init__(metadata)
        self.should_fail = should_fail
    
    async def execute(self, context: ToolExecutionContext) -> ToolResult:
        """Execute the mock tool."""
        if self.should_fail:
            return ToolResult(
                success=False,
                error="Mock tool failed",
                tool_name=self.metadata.name,
                parameters=context.parameters,
                status=ToolStatus.FAILED
            )
        
        return ToolResult(
            success=True,
            result={"mock_result": "success", "parameters": context.parameters},
            tool_name=self.metadata.name,
            parameters=context.parameters,
            status=ToolStatus.COMPLETED
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        return isinstance(parameters, dict)


class TestToolInterface:
    """Test cases for ToolInterface."""
    
    def test_tool_interface_creation(self):
        """Test tool interface creation."""
        metadata = ToolMetadata(
            name="mock_tool",
            description="Mock tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata)
        
        assert tool.metadata.name == "mock_tool"
        assert tool.metadata.description == "Mock tool"
        assert tool.metadata.category == ToolCategory.UTILITY
    
    def test_tool_execution_success(self):
        """Test successful tool execution."""
        metadata = ToolMetadata(
            name="success_tool",
            description="Success tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata, should_fail=False)
        context = ToolExecutionContext(
            execution_id="test_exec",
            tool_name="success_tool",
            parameters={"test": "value"}
        )
        
        result = asyncio.run(tool.execute(context))
        
        assert result.success is True
        assert result.result == {"mock_result": "success", "parameters": {"test": "value"}}
        assert result.tool_name == "success_tool"
        assert result.status == ToolStatus.COMPLETED
    
    def test_tool_execution_failure(self):
        """Test failed tool execution."""
        metadata = ToolMetadata(
            name="fail_tool",
            description="Fail tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata, should_fail=True)
        context = ToolExecutionContext(
            execution_id="test_exec",
            tool_name="fail_tool",
            parameters={"test": "value"}
        )
        
        result = asyncio.run(tool.execute(context))
        
        assert result.success is False
        assert result.error == "Mock tool failed"
        assert result.tool_name == "fail_tool"
        assert result.status == ToolStatus.FAILED
    
    def test_tool_stats_update(self):
        """Test tool statistics update."""
        metadata = ToolMetadata(
            name="stats_tool",
            description="Stats tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata)
        
        # Update stats
        tool.update_stats(1.5, True)
        
        assert tool.metadata.execution_count == 1
        assert tool.metadata.success_count == 1
        assert tool.metadata.failure_count == 0
        assert tool.metadata.average_execution_time == 1.5
        
        # Update with failure
        tool.update_stats(0.5, False)
        
        assert tool.metadata.execution_count == 2
        assert tool.metadata.success_count == 1
        assert tool.metadata.failure_count == 1
        assert tool.metadata.average_execution_time == 1.0


class TestFunctionTool:
    """Test cases for FunctionTool."""
    
    def test_sync_function_tool(self):
        """Test function tool with sync function."""
        def test_function(input_params):
            return input_params["input"] * 2
        
        metadata = ToolMetadata(
            name="sync_tool",
            description="Sync tool",
            category=ToolCategory.UTILITY
        )
        
        tool = FunctionTool(test_function, metadata)
        context = ToolExecutionContext(
            execution_id="test_exec",
            tool_name="sync_tool",
            parameters={"input": 21}
        )
        
        result = asyncio.run(tool.execute(context))
        
        assert result.success is True
        assert result.result == {"result": 42}
        assert result.tool_name == "sync_tool"
    
    def test_async_function_tool(self):
        """Test function tool with async function."""
        async def test_function(input_params):
            await asyncio.sleep(0.01)
            return input_params["input"] * 3
        
        metadata = ToolMetadata(
            name="async_tool",
            description="Async tool",
            category=ToolCategory.UTILITY
        )
        
        tool = FunctionTool(test_function, metadata)
        context = ToolExecutionContext(
            execution_id="test_exec",
            tool_name="async_tool",
            parameters={"input": 14}
        )
        
        result = asyncio.run(tool.execute(context))
        
        assert result.success is True
        assert result.result == {"result": 42}
        assert result.tool_name == "async_tool"
    
    def test_function_tool_with_exception(self):
        """Test function tool with exception."""
        def test_function(input_params):
            raise ValueError("Test exception")
        
        metadata = ToolMetadata(
            name="exception_tool",
            description="Exception tool",
            category=ToolCategory.UTILITY
        )
        
        tool = FunctionTool(test_function, metadata)
        context = ToolExecutionContext(
            execution_id="test_exec",
            tool_name="exception_tool",
            parameters={"input": 42}
        )
        
        result = asyncio.run(tool.execute(context))
        
        assert result.success is False
        assert "Test exception" in result.error
        assert result.tool_name == "exception_tool"
        assert result.status == ToolStatus.FAILED
    
    def test_function_tool_parameter_validation(self):
        """Test function tool parameter validation."""
        def test_function(input_params):
            return input_params["required_param"]
        
        metadata = ToolMetadata(
            name="validation_tool",
            description="Validation tool",
            category=ToolCategory.UTILITY
        )
        
        tool = FunctionTool(test_function, metadata)
        
        # Valid parameters (function will work with these)
        assert tool.validate_parameters({"required_param": "value"}) is True
        
        # Test with multiple parameter function
        def multi_param_function(param1, param2):
            return param1 + param2
        
        multi_tool = FunctionTool(multi_param_function, ToolMetadata(
            name="multi_validation_tool",
            description="Multi param validation tool",
            category=ToolCategory.UTILITY
        ))
        
        # Valid parameters for multi-param function
        assert multi_tool.validate_parameters({"param1": "a", "param2": "b"}) is True
        
        # Invalid parameters for multi-param function (missing param2)
        assert multi_tool.validate_parameters({"param1": "a"}) is False
    
    def test_tool_decorator(self):
        """Test tool decorator."""
        @tool(
            name="decorated_tool",
            description="Decorated tool",
            category=ToolCategory.UTILITY
        )
        def decorated_function(params):
            return {"decorated": True, "input": params["input"]}
        
        # Check that tool instance was created
        assert hasattr(decorated_function, '_tool_instance')
        tool_instance = decorated_function._tool_instance
        
        assert tool_instance.metadata.name == "decorated_tool"
        assert tool_instance.metadata.description == "Decorated tool"
        assert tool_instance.metadata.category == ToolCategory.UTILITY


class TestToolRegistry:
    """Test cases for ToolRegistry."""
    
    def test_tool_registration(self):
        """Test tool registration."""
        registry = ToolRegistry()
        
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.UTILITY,
            tags=["test", "utility"]
        )
        
        tool = MockTool(metadata)
        
        # Register tool
        result = registry.register_tool(tool)
        assert result is True
        
        # Check that tool is registered
        assert registry.get_tool("test_tool") == tool
        assert len(registry.get_all_tools()) == 1
    
    def test_tool_unregistration(self):
        """Test tool unregistration."""
        registry = ToolRegistry()
        
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata)
        registry.register_tool(tool)
        
        # Unregister tool
        result = registry.unregister_tool("test_tool")
        assert result is True
        
        # Check that tool is unregistered
        assert registry.get_tool("test_tool") is None
        assert len(registry.get_all_tools()) == 0
    
    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        
        # Create tools in different categories
        utility_tool = MockTool(ToolMetadata(
            name="utility_tool",
            description="Utility tool",
            category=ToolCategory.UTILITY
        ))
        
        system_tool = MockTool(ToolMetadata(
            name="system_tool",
            description="System tool",
            category=ToolCategory.SYSTEM
        ))
        
        registry.register_tool(utility_tool)
        registry.register_tool(system_tool)
        
        # Get tools by category
        utility_tools = registry.get_tools_by_category(ToolCategory.UTILITY)
        system_tools = registry.get_tools_by_category(ToolCategory.SYSTEM)
        
        assert len(utility_tools) == 1
        assert len(system_tools) == 1
        assert utility_tools[0].metadata.name == "utility_tool"
        assert system_tools[0].metadata.name == "system_tool"
    
    def test_get_tools_by_tag(self):
        """Test getting tools by tag."""
        registry = ToolRegistry()
        
        # Create tools with different tags
        tool1 = MockTool(ToolMetadata(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.UTILITY,
            tags=["test", "tag1"]
        ))
        
        tool2 = MockTool(ToolMetadata(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.UTILITY,
            tags=["test", "tag2"]
        ))
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Get tools by tag
        test_tools = registry.get_tools_by_tag("test")
        tag1_tools = registry.get_tools_by_tag("tag1")
        
        assert len(test_tools) == 2
        assert len(tag1_tools) == 1
        assert tag1_tools[0].metadata.name == "tool1"
    
    def test_search_tools(self):
        """Test searching tools."""
        registry = ToolRegistry()
        
        # Create tools
        tool1 = MockTool(ToolMetadata(
            name="calculate_tool",
            description="Calculate mathematical expressions",
            category=ToolCategory.UTILITY,
            tags=["math", "calculation"]
        ))
        
        tool2 = MockTool(ToolMetadata(
            name="format_tool",
            description="Format text content",
            category=ToolCategory.UTILITY,
            tags=["text", "formatting"]
        ))
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Search tools
        math_tools = registry.search_tools("math")
        text_tools = registry.search_tools("text")
        
        assert len(math_tools) == 1
        assert len(text_tools) == 1
        assert math_tools[0].metadata.name == "calculate_tool"
        assert text_tools[0].metadata.name == "format_tool"
    
    def test_get_tool_stats(self):
        """Test getting tool statistics."""
        registry = ToolRegistry()
        
        # Create and register tools
        tool1 = MockTool(ToolMetadata(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.UTILITY
        ))
        
        tool2 = MockTool(ToolMetadata(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.SYSTEM
        ))
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Get stats
        stats = registry.get_tool_stats()
        
        assert stats["total_tools"] == 2
        assert stats["active_tools"] == 2
        assert stats["categories"]["utility"] == 1
        assert stats["categories"]["system"] == 1


class TestToolExecutor:
    """Test cases for ToolExecutor."""
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # Create and register tool
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata)
        registry.register_tool(tool)
        
        # Execute tool
        result = asyncio.run(executor.execute_tool(
            "test_tool",
            {"test": "value"}
        ))
        
        assert result.success is True
        assert result.result == {"mock_result": "success", "parameters": {"test": "value"}}
        assert result.tool_name == "test_tool"
    
    def test_execute_tool_not_found(self):
        """Test execution of non-existent tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # Execute non-existent tool
        result = asyncio.run(executor.execute_tool(
            "nonexistent_tool",
            {"test": "value"}
        ))
        
        assert result.success is False
        assert "not found" in result.error
        assert result.tool_name == "nonexistent_tool"
    
    def test_execute_tool_timeout(self):
        """Test tool execution timeout."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # Create slow tool
        async def slow_function(params):
            await asyncio.sleep(0.1)
            return {"result": "slow"}
        
        metadata = ToolMetadata(
            name="slow_tool",
            description="Slow tool",
            category=ToolCategory.UTILITY,
            timeout=0.05  # Very short timeout
        )
        
        tool = FunctionTool(slow_function, metadata)
        registry.register_tool(tool)
        
        # Execute tool (should timeout)
        result = asyncio.run(executor.execute_tool(
            "slow_tool",
            {"test": "value"}
        ))
        
        assert result.success is False
        assert "timed out" in result.error
        assert result.status == ToolStatus.TIMEOUT
    
    def test_execute_tools_parallel(self):
        """Test parallel tool execution."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # Create multiple tools
        for i in range(3):
            metadata = ToolMetadata(
                name=f"tool_{i}",
                description=f"Tool {i}",
                category=ToolCategory.UTILITY
            )
            
            tool = MockTool(metadata)
            registry.register_tool(tool)
        
        # Execute tools in parallel
        tool_calls = [
            {"tool_name": "tool_0", "parameters": {"test": "value0"}},
            {"tool_name": "tool_1", "parameters": {"test": "value1"}},
            {"tool_name": "tool_2", "parameters": {"test": "value2"}}
        ]
        
        results = asyncio.run(executor.execute_tools_parallel(tool_calls))
        
        assert len(results) == 3
        for result in results:
            assert result.success is True
    
    def test_execute_tool_chain(self):
        """Test tool chain execution."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # Create tools for chain
        def add_tool(params):
            return {"result": params["a"] + params["b"]}
        
        def multiply_tool(params):
            return {"result": params["result"] * 2}  # Use previous result
        
        metadata1 = ToolMetadata(
            name="add_tool",
            description="Add tool",
            category=ToolCategory.UTILITY
        )
        
        metadata2 = ToolMetadata(
            name="multiply_tool",
            description="Multiply tool",
            category=ToolCategory.UTILITY
        )
        
        tool1 = FunctionTool(add_tool, metadata1)
        tool2 = FunctionTool(multiply_tool, metadata2)
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Execute chain
        chain = [
            {"tool_name": "add_tool", "parameters": {"a": 10, "b": 5}},
            {"tool_name": "multiply_tool", "parameters": {}}
        ]
        
        result = asyncio.run(executor.execute_tool_chain(chain))
        
        assert result["success"] is True
        assert result["final_context"]["result"] == 30  # (10 + 5) * 2
    
    def test_execution_history(self):
        """Test execution history tracking."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # Create and register tool
        metadata = ToolMetadata(
            name="history_tool",
            description="History tool",
            category=ToolCategory.UTILITY
        )
        
        tool = MockTool(metadata)
        registry.register_tool(tool)
        
        # Execute tool multiple times
        for i in range(3):
            asyncio.run(executor.execute_tool(
                "history_tool",
                {"test": f"value{i}"}
            ))
        
        # Get history
        history = executor.get_execution_history()
        
        assert len(history) == 3
        assert history[0]["tool_name"] == "history_tool"
        
        # Get history for specific tool
        tool_history = executor.get_execution_history("history_tool")
        assert len(tool_history) == 3
        
        # Get history for non-existent tool
        empty_history = executor.get_execution_history("nonexistent_tool")
        assert len(empty_history) == 0


class TestToolManager:
    """Test cases for ToolManager."""
    
    def test_tool_manager_initialization(self):
        """Test tool manager initialization."""
        config = ContextManagerConfig(max_workers=3)
        manager = ToolManager(config)
        
        assert manager.config == config
        assert len(manager.registry.tools) > 0  # Should have built-in tools
        assert manager.executor.max_workers == 3
    
    def test_builtin_tools_registration(self):
        """Test that built-in tools are registered."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        # Check that built-in tools are registered
        assert manager.get_tool("echo") is not None
        assert manager.get_tool("get_system_info") is not None
        assert manager.get_tool("calculate") is not None
        assert manager.get_tool("format_text") is not None
    
    def test_list_tools(self):
        """Test listing tools."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        # List all tools
        all_tools = manager.list_tools()
        assert len(all_tools) >= 4  # At least built-in tools
        
        # List tools by category
        utility_tools = manager.list_tools(category=ToolCategory.UTILITY)
        assert len(utility_tools) >= 3
        
        # List tools by tag
        tagged_tools = manager.list_tools(tag="utility")
        assert len(tagged_tools) >= 1
    
    def test_search_tools(self):
        """Test searching tools."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        # Search for tools
        math_tools = manager.search_tools("math")
        assert len(math_tools) >= 1
        
        system_tools = manager.search_tools("system")
        assert len(system_tools) >= 1
    
    def test_execute_tool_request(self):
        """Test executing tool with request."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        request = ToolExecutionRequest(
            tool_name="echo",
            parameters={"message": "Hello, World!"},
            user_id="test_user",
            session_id="test_session"
        )
        
        response = asyncio.run(manager.execute_tool(request))
        
        assert response.success is True
        assert response.result["echo"]["message"] == "Hello, World!"
        assert response.tool_name == "echo"
        assert response.execution_id is not None
    
    def test_execute_tools_parallel(self):
        """Test parallel execution with manager."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        requests = [
            ToolExecutionRequest(
                tool_name="echo",
                parameters={"message": "Message 1"}
            ),
            ToolExecutionRequest(
                tool_name="echo",
                parameters={"message": "Message 2"}
            )
        ]
        
        responses = asyncio.run(manager.execute_tools_parallel(requests))
        
        assert len(responses) == 2
        for response in responses:
            assert response.success is True
            assert response.tool_name == "echo"
    
    def test_execute_tool_chain(self):
        """Test tool chain execution with manager."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        requests = [
            ToolExecutionRequest(
                tool_name="calculate",
                parameters={"expression": "10 + 5"}
            ),
            ToolExecutionRequest(
                tool_name="calculate",
                parameters={"expression": "15 * 2"}  # Direct calculation instead of using result
            )
        ]
        
        result = asyncio.run(manager.execute_tool_chain(requests))
        
        assert result["success"] is True
        # Final result should be 30
        assert result["chain_results"][-1]["result"]["result"] == 30
    
    def test_get_tool_stats(self):
        """Test getting tool statistics."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        # Execute some tools
        request = ToolExecutionRequest(
            tool_name="echo",
            parameters={"message": "Test"}
        )
        
        asyncio.run(manager.execute_tool(request))
        
        # Get stats
        stats = manager.get_tool_stats()
        
        assert "performance_metrics" in stats
        assert "registry_stats" in stats
        assert stats["performance_metrics"]["total_executions"] >= 1
        assert stats["registry_stats"]["total_tools"] >= 4
    
    def test_health_check(self):
        """Test health check."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        health = manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["registered_tools"] >= 4
        assert "performance" in health
        assert "success_rate" in health["performance"]
    
    def test_get_active_executions(self):
        """Test getting active executions."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        # Start an execution (this is simplified)
        request = ToolExecutionRequest(
            tool_name="echo",
            parameters={"message": "Test"}
        )
        
        response = asyncio.run(manager.execute_tool(request))
        
        # Get active executions (should be empty after completion)
        active = manager.get_active_executions()
        assert len(active) == 0
    
    def test_get_execution_history(self):
        """Test getting execution history."""
        config = ContextManagerConfig()
        manager = ToolManager(config)
        
        # Execute some tools
        for i in range(2):
            request = ToolExecutionRequest(
                tool_name="echo",
                parameters={"message": f"Test {i}"}
            )
            asyncio.run(manager.execute_tool(request))
        
        # Get history
        history = manager.get_execution_history()
        assert len(history) >= 2
        
        # Get history for specific tool
        echo_history = manager.get_execution_history("echo")
        assert len(echo_history) >= 2


class TestToolExecutionRequest:
    """Test cases for ToolExecutionRequest."""
    
    def test_request_creation(self):
        """Test basic request creation."""
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={"param1": "value1"}
        )
        
        assert request.tool_name == "test_tool"
        assert request.parameters == {"param1": "value1"}
        assert request.execution_id is None
        assert request.user_id is None
        assert request.session_id is None
        assert request.priority == "normal"
    
    def test_request_with_all_fields(self):
        """Test request with all fields."""
        request = ToolExecutionRequest(
            tool_name="complex_tool",
            parameters={"param1": "value1", "param2": 42},
            execution_id="exec_123",
            user_id="user_456",
            session_id="session_789",
            priority="high",
            timeout=60.0,
            context={"custom": "data"}
        )
        
        assert request.tool_name == "complex_tool"
        assert request.parameters == {"param1": "value1", "param2": 42}
        assert request.execution_id == "exec_123"
        assert request.user_id == "user_456"
        assert request.session_id == "session_789"
        assert request.priority == "high"
        assert request.timeout == 60.0
        assert request.context == {"custom": "data"}


class TestToolExecutionResponse:
    """Test cases for ToolExecutionResponse."""
    
    def test_success_response(self):
        """Test successful response."""
        response = ToolExecutionResponse(
            success=True,
            result={"test": "data"},
            execution_id="exec_123",
            tool_name="test_tool",
            execution_time=0.5
        )
        
        assert response.success is True
        assert response.result == {"test": "data"}
        assert response.execution_id == "exec_123"
        assert response.tool_name == "test_tool"
        assert response.execution_time == 0.5
        assert response.error is None
        assert response.status == ToolStatus.COMPLETED
    
    def test_failure_response(self):
        """Test failed response."""
        response = ToolExecutionResponse(
            success=False,
            result=None,
            execution_id="exec_123",
            tool_name="test_tool",
            execution_time=0.1,
            error="Tool execution failed",
            status=ToolStatus.FAILED
        )
        
        assert response.success is False
        assert response.result is None
        assert response.execution_id == "exec_123"
        assert response.tool_name == "test_tool"
        assert response.execution_time == 0.1
        assert response.error == "Tool execution failed"
        assert response.status == ToolStatus.FAILED