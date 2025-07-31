"""
Tests for core data models.
"""

import pytest
import time
import uuid
from datetime import datetime, timedelta
from context_manager.core.models import (
    Message, ContextWindow, Memory, ToolResult, SystemState,
    ContextPriority, MemoryType, ToolStatus
)


class TestMessage:
    """Test cases for Message class."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = Message(
            role="user",
            content="Hello, world!"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.priority == ContextPriority.MEDIUM
        assert isinstance(message.timestamp, float)
        assert isinstance(message.message_id, str)
        assert message.token_count is not None
        assert message.metadata == {}
    
    def test_message_with_custom_priority(self):
        """Test message with custom priority."""
        message = Message(
            role="assistant",
            content="Response",
            priority=ContextPriority.HIGH
        )
        
        assert message.priority == ContextPriority.HIGH
    
    def test_message_validation(self):
        """Test message validation."""
        with pytest.raises(ValueError, match="Message role cannot be empty"):
            Message(role="", content="test")
        
        with pytest.raises(ValueError, match="Message content must be a string"):
            Message(role="user", content=123)
    
    def test_message_to_dict(self):
        """Test message serialization."""
        message = Message(
            role="user",
            content="Test message",
            priority=ContextPriority.CRITICAL,
            metadata={"key": "value"}
        )
        
        data = message.to_dict()
        
        assert data['role'] == "user"
        assert data['content'] == "Test message"
        assert data['priority'] == "critical"
        assert data['metadata'] == {"key": "value"}
        assert 'message_id' in data
        assert 'timestamp' in data
    
    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            'role': 'assistant',
            'content': 'Response message',
            'priority': 'high',
            'metadata': {'response_id': '123'},
            'message_id': str(uuid.uuid4()),
            'timestamp': time.time()
        }
        
        message = Message.from_dict(data)
        
        assert message.role == "assistant"
        assert message.content == "Response message"
        assert message.priority == ContextPriority.HIGH
        assert message.metadata == {'response_id': '123'}
    
    def test_message_copy(self):
        """Test message copying."""
        original = Message(
            role="user",
            content="Original",
            metadata={"key": "value"}
        )
        
        copy = original.copy()
        
        assert copy.role == original.role
        assert copy.content == original.content
        assert copy.message_id == original.message_id
        assert copy.metadata == original.metadata
        assert copy is not original  # Different objects
    
    def test_message_expiration(self):
        """Test message expiration check."""
        message = Message(role="user", content="test")
        
        # Should not be expired immediately
        assert not message.is_expired(1.0)
        
        # Should be expired after waiting
        time.sleep(0.1)
        assert message.is_expired(0.05)  # 50ms max age


class TestContextWindow:
    """Test cases for ContextWindow class."""
    
    def test_context_window_creation(self):
        """Test basic context window creation."""
        context = ContextWindow(max_tokens=1000)
        
        assert context.max_tokens == 1000
        assert context.current_tokens == 0
        assert len(context.messages) == 0
        assert context.compression_ratio == 0.8
    
    def test_context_window_validation(self):
        """Test context window validation."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ContextWindow(max_tokens=0)
        
        with pytest.raises(ValueError, match="compression_ratio must be between 0 and 1"):
            ContextWindow(compression_ratio=1.5)
    
    def test_add_message(self):
        """Test adding messages to context window."""
        context = ContextWindow(max_tokens=1000)
        message = Message(role="user", content="Hello")
        
        result = context.add_message(message)
        
        assert result is True
        assert len(context.messages) == 1
        assert context.current_tokens > 0
        assert context.messages[0] == message
    
    def test_remove_message(self):
        """Test removing messages from context window."""
        context = ContextWindow(max_tokens=1000)
        message = Message(role="user", content="Hello")
        context.add_message(message)
        
        result = context.remove_message(message.message_id)
        
        assert result is True
        assert len(context.messages) == 0
        assert context.current_tokens == 0
    
    def test_get_messages_by_priority(self):
        """Test filtering messages by priority."""
        context = ContextWindow(max_tokens=1000)
        
        # Add messages with different priorities
        high_msg = Message(role="user", content="Important", priority=ContextPriority.HIGH)
        low_msg = Message(role="user", content="Normal", priority=ContextPriority.LOW)
        
        context.add_message(high_msg)
        context.add_message(low_msg)
        
        high_priority_msgs = context.get_messages_by_priority(ContextPriority.HIGH)
        low_priority_msgs = context.get_messages_by_priority(ContextPriority.LOW)
        
        assert len(high_priority_msgs) == 1
        assert len(low_priority_msgs) == 1
        assert high_priority_msgs[0] == high_msg
        assert low_priority_msgs[0] == low_msg
    
    def test_context_window_utilization(self):
        """Test context window utilization calculation."""
        context = ContextWindow(max_tokens=1000)
        
        # Add a message with known token count
        message = Message(role="user", content="Hello world", token_count=50)
        context.add_message(message)
        
        assert context.get_utilization() == 0.05  # 50/1000 = 0.05
        assert context.is_full() is False
    
    def test_context_window_copy(self):
        """Test context window copying."""
        original = ContextWindow(max_tokens=500)
        message = Message(role="user", content="test")
        original.add_message(message)
        
        copy = original.copy()
        
        assert copy.max_tokens == original.max_tokens
        assert len(copy.messages) == len(original.messages)
        assert copy.current_tokens == original.current_tokens
        assert copy is not original
        assert copy.messages[0] is not original.messages[0]  # Deep copy


class TestMemory:
    """Test cases for Memory class."""
    
    def test_memory_creation(self):
        """Test basic memory creation."""
        memory = Memory(
            content="This is a memory",
            importance=0.8,
            tags=["important", "test"]
        )
        
        assert memory.content == "This is a memory"
        assert memory.importance == 0.8
        assert memory.tags == ["important", "test"]
        assert memory.memory_type == MemoryType.SHORT_TERM
        assert memory.access_count == 0
        assert isinstance(memory.timestamp, datetime)
    
    def test_memory_validation(self):
        """Test memory validation."""
        with pytest.raises(ValueError, match="Memory content must be a string"):
            Memory(content=123)
        
        with pytest.raises(ValueError, match="Importance must be between 0.0 and 1.0"):
            Memory(content="test", importance=1.5)
    
    def test_memory_access_update(self):
        """Test memory access statistics update."""
        memory = Memory(content="test")
        original_count = memory.access_count
        original_accessed = memory.last_accessed
        
        time.sleep(0.01)  # Small delay
        memory.update_access()
        
        assert memory.access_count == original_count + 1
        assert memory.last_accessed > original_accessed
    
    def test_memory_age_calculation(self):
        """Test memory age calculation."""
        memory = Memory(content="test")
        
        # Should have some age
        age = memory.get_age()
        assert age >= 0
        
        # Should increase over time
        time.sleep(0.01)
        assert memory.get_age() > age
    
    def test_memory_scoring(self):
        """Test memory score calculation."""
        # High importance, recent memory
        high_score_memory = Memory(
            content="important recent",
            importance=0.9,
            memory_type=MemoryType.SHORT_TERM
        )
        high_score_memory.update_access()
        
        # Low importance, old memory
        low_score_memory = Memory(
            content="unimportant old",
            importance=0.1,
            memory_type=MemoryType.LONG_TERM
        )
        
        high_score = high_score_memory.calculate_score()
        low_score = low_score_memory.calculate_score()
        
        assert high_score > low_score
    
    def test_memory_serialization(self):
        """Test memory serialization and deserialization."""
        original = Memory(
            content="test memory",
            importance=0.7,
            tags=["test", "serialization"],
            metadata={"key": "value"}
        )
        original.update_access()
        
        # Serialize
        data = original.to_dict()
        
        # Deserialize
        restored = Memory.from_dict(data)
        
        assert restored.content == original.content
        assert restored.importance == original.importance
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata
        assert restored.access_count == original.access_count


class TestToolResult:
    """Test cases for ToolResult class."""
    
    def test_successful_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            result="Operation completed",
            execution_time=0.5,
            tool_name="test_tool"
        )
        
        assert result.success is True
        assert result.result == "Operation completed"
        assert result.error is None
        assert result.execution_time == 0.5
        assert result.tool_name == "test_tool"
        assert result.status == ToolStatus.COMPLETED
    
    def test_failed_result(self):
        """Test failed tool result."""
        result = ToolResult(
            success=False,
            error="Operation failed",
            execution_time=0.1,
            tool_name="failing_tool",
            status=ToolStatus.FAILED
        )
        
        assert result.success is False
        assert result.error == "Operation failed"
        assert result.result is None
        assert result.status == ToolStatus.FAILED
    
    def test_tool_result_validation(self):
        """Test tool result validation."""
        # Successful result with error should fail
        with pytest.raises(ValueError, match="Successful execution cannot have error"):
            ToolResult(success=True, error="Some error")
        
        # Failed result without error should fail
        with pytest.raises(ValueError, match="Failed execution must have error"):
            ToolResult(success=False)
        
        # Negative execution time should fail
        with pytest.raises(ValueError, match="Execution time cannot be negative"):
            ToolResult(success=True, result="test", execution_time=-1.0)
    
    def test_tool_result_serialization(self):
        """Test tool result serialization."""
        original = ToolResult(
            success=True,
            result={"data": "value"},
            execution_time=1.5,
            tool_name="serializer_test",
            parameters={"param1": "value1"}
        )
        
        data = original.to_dict()
        restored = ToolResult.from_dict(data)
        
        assert restored.success == original.success
        assert restored.result == original.result
        assert restored.execution_time == original.execution_time
        assert restored.tool_name == original.tool_name
        assert restored.parameters == original.parameters


class TestSystemState:
    """Test cases for SystemState class."""
    
    def test_system_state_creation(self):
        """Test basic system state creation."""
        state = SystemState()
        
        assert isinstance(state.context_window, ContextWindow)
        assert state.memory_stats == {}
        assert state.tool_stats == {}
        assert state.performance_metrics == {}
        assert isinstance(state.last_updated, float)
    
    def test_system_state_with_data(self):
        """Test system state with data."""
        context = ContextWindow(max_tokens=2000)
        memory_stats = {"total_memories": 100, "used_memory": 50}
        tool_stats = {"tools_executed": 25, "success_rate": 0.8}
        
        state = SystemState(
            context_window=context,
            memory_stats=memory_stats,
            tool_stats=tool_stats
        )
        
        assert state.context_window == context
        assert state.memory_stats == memory_stats
        assert state.tool_stats == tool_stats
    
    def test_system_state_serialization(self):
        """Test system state serialization."""
        context = ContextWindow(max_tokens=1000)
        message = Message(role="user", content="test")
        context.add_message(message)
        
        state = SystemState(
            context_window=context,
            memory_stats={"count": 10},
            tool_stats={"executed": 5}
        )
        
        data = state.to_dict()
        
        assert 'context_window' in data
        assert 'memory_stats' in data
        assert 'tool_stats' in data
        assert 'last_updated' in data
        assert data['context_window']['message_count'] == 1
        assert data['context_window']['token_count'] > 0