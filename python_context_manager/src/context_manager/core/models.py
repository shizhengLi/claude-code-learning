"""
Core data models for the context manager system.

This module defines the fundamental data structures used throughout the system,
including messages, contexts, memories, and tools.
"""

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json


class ContextPriority(Enum):
    """Priority levels for context messages."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class ToolStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Message:
    """
    Represents a message in the conversation context.
    
    Attributes:
        role: The role of the message sender (user, assistant, system, etc.)
        content: The content of the message
        timestamp: When the message was created
        priority: Priority level for context management
        metadata: Additional metadata associated with the message
        message_id: Unique identifier for the message
        token_count: Estimated token count for the message
    """
    
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    priority: ContextPriority = ContextPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_count: Optional[int] = None
    
    def __post_init__(self):
        """Validate and initialize message data."""
        if not self.role:
            raise ValueError("Message role cannot be empty")
            
        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string")
            
        # Estimate token count if not provided
        if self.token_count is None:
            self.token_count = self._estimate_tokens()
            
    def _estimate_tokens(self) -> int:
        """
        Estimate token count for the message.
        
        Returns:
            Estimated token count.
        """
        # Simple token estimation based on words and characters
        words = self.content.split()
        word_tokens = len(words) * 1.3  # Average tokens per word
        
        # Add overhead for message structure
        structure_tokens = 4  # For role, timestamp, etc.
        
        # Add metadata tokens
        metadata_tokens = len(json.dumps(self.metadata)) * 0.5 if self.metadata else 0
        
        return int(word_tokens + structure_tokens + metadata_tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'message_id': self.message_id,
            'token_count': self.token_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary representation."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data.get('timestamp', time.time()),
            priority=ContextPriority(data.get('priority', 'medium')),
            metadata=data.get('metadata', {}),
            message_id=data.get('message_id', str(uuid.uuid4())),
            token_count=data.get('token_count'),
        )
    
    def is_expired(self, max_age: float) -> bool:
        """
        Check if message is expired based on age.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            True if message is expired
        """
        return time.time() - self.timestamp > max_age
    
    def copy(self) -> 'Message':
        """Create a copy of the message."""
        return Message(
            role=self.role,
            content=self.content,
            timestamp=self.timestamp,
            priority=self.priority,
            metadata=self.metadata.copy(),
            message_id=self.message_id,
            token_count=self.token_count,
        )


@dataclass
class ContextWindow:
    """
    Represents the current context window with messages and metadata.
    
    Attributes:
        messages: List of messages in the context
        max_tokens: Maximum token limit for the context
        current_tokens: Current token count in the context
        compression_ratio: Target compression ratio
        metadata: Additional context metadata
        created_at: When the context window was created
        last_updated: When the context window was last updated
    """
    
    messages: List[Message] = field(default_factory=list)
    max_tokens: int = 4000
    current_tokens: int = 0
    compression_ratio: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate context window configuration."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
            
        if not 0 < self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be between 0 and 1")
            
        # Recalculate current tokens
        self.current_tokens = sum(msg.token_count or 0 for msg in self.messages)
    
    def add_message(self, message: Message) -> bool:
        """
        Add a message to the context window.
        
        Args:
            message: Message to add
            
        Returns:
            True if message was added successfully
        """
        self.messages.append(message)
        self.current_tokens += message.token_count or 0
        self.last_updated = time.time()
        return True
    
    def remove_message(self, message_id: str) -> bool:
        """
        Remove a message from the context window.
        
        Args:
            message_id: ID of message to remove
            
        Returns:
            True if message was removed successfully
        """
        for i, message in enumerate(self.messages):
            if message.message_id == message_id:
                self.current_tokens -= message.token_count or 0
                self.messages.pop(i)
                self.last_updated = time.time()
                return True
        return False
    
    def get_messages_by_priority(self, priority: ContextPriority) -> List[Message]:
        """
        Get messages with specified priority.
        
        Args:
            priority: Priority level to filter by
            
        Returns:
            List of messages with the specified priority
        """
        return [msg for msg in self.messages if msg.priority == priority]
    
    def get_total_tokens(self) -> int:
        """Get total token count in the context window."""
        return sum(msg.token_count or 0 for msg in self.messages)
    
    def is_full(self) -> bool:
        """Check if context window is at capacity."""
        return self.current_tokens >= self.max_tokens
    
    def get_utilization(self) -> float:
        """Get context window utilization ratio."""
        return self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
    
    def clear(self) -> None:
        """Clear all messages from the context window."""
        self.messages.clear()
        self.current_tokens = 0
        self.last_updated = time.time()
    
    def copy(self) -> 'ContextWindow':
        """Create a copy of the context window."""
        return ContextWindow(
            messages=[msg.copy() for msg in self.messages],
            max_tokens=self.max_tokens,
            current_tokens=self.current_tokens,
            compression_ratio=self.compression_ratio,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
            last_updated=self.last_updated,
        )


@dataclass
class Memory:
    """
    Represents a memory unit in the memory system.
    
    Attributes:
        content: The content of the memory
        timestamp: When the memory was created
        importance: Importance score (0.0 to 1.0)
        tags: List of tags for categorization
        metadata: Additional metadata
        memory_id: Unique identifier for the memory
        memory_type: Type of memory storage
        access_count: Number of times this memory has been accessed
        last_accessed: When this memory was last accessed
    """
    
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.SHORT_TERM
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate memory data."""
        if not isinstance(self.content, str):
            raise ValueError("Memory content must be a string")
            
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
            
        if not isinstance(self.tags, list):
            raise ValueError("Tags must be a list")
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age(self) -> float:
        """Get age of memory in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    
    def get_access_frequency(self) -> float:
        """Get access frequency (accesses per hour)."""
        age_hours = self.get_age() / 3600
        return self.access_count / age_hours if age_hours > 0 else 0.0
    
    def calculate_score(self) -> float:
        """
        Calculate overall memory score for ranking.
        
        Returns:
            Combined score based on importance, recency, and frequency
        """
        importance_score = self.importance * 0.5
        
        # Recency score (more recent = higher score)
        age_score = max(0, 1 - (self.get_age() / (30 * 24 * 3600))) * 0.3  # 30 day decay
        
        # Frequency score
        frequency_score = min(1.0, self.get_access_frequency() / 10) * 0.2
        
        return importance_score + age_score + frequency_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation."""
        return {
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'tags': self.tags,
            'metadata': self.metadata,
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary representation."""
        return cls(
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            importance=data.get('importance', 0.5),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            memory_id=data.get('memory_id', str(uuid.uuid4())),
            memory_type=MemoryType(data.get('memory_type', 'short_term')),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data.get('last_accessed', datetime.now().isoformat())),
        )


@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.
    
    Attributes:
        success: Whether the tool execution was successful
        result: The result data (if successful)
        error: Error message (if failed)
        execution_time: Time taken to execute the tool
        execution_id: Unique identifier for the execution
        tool_name: Name of the tool that was executed
        parameters: Parameters used for the execution
        status: Status of the execution
        metadata: Additional metadata
    """
    
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: ToolStatus = ToolStatus.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate tool result."""
        if self.success and self.error:
            raise ValueError("Successful execution cannot have error")
            
        if not self.success and not self.error:
            raise ValueError("Failed execution must have error")
            
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool result to dictionary representation."""
        return {
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'execution_id': self.execution_id,
            'tool_name': self.tool_name,
            'parameters': self.parameters,
            'status': self.status.value,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """Create tool result from dictionary representation."""
        return cls(
            success=data['success'],
            result=data.get('result'),
            error=data.get('error'),
            execution_time=data.get('execution_time', 0.0),
            execution_id=data.get('execution_id', str(uuid.uuid4())),
            tool_name=data.get('tool_name', ''),
            parameters=data.get('parameters', {}),
            status=ToolStatus(data.get('status', 'completed')),
            metadata=data.get('metadata', {}),
        )


@dataclass
class SystemState:
    """
    Represents the overall system state.
    
    Attributes:
        context_window: Current context window
        memory_stats: Memory system statistics
        tool_stats: Tool system statistics
        performance_metrics: Performance metrics
        system_info: System information
        last_updated: When the state was last updated
    """
    
    context_window: ContextWindow = field(default_factory=ContextWindow)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    tool_stats: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    
    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system state to dictionary representation."""
        return {
            'context_window': {
                'message_count': len(self.context_window.messages),
                'token_count': self.context_window.current_tokens,
                'max_tokens': self.context_window.max_tokens,
                'utilization': self.context_window.get_utilization(),
            },
            'memory_stats': self.memory_stats,
            'tool_stats': self.tool_stats,
            'performance_metrics': self.performance_metrics,
            'system_info': self.system_info,
            'last_updated': self.last_updated,
        }