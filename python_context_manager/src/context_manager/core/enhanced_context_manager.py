"""
Enhanced ContextManager with full system integration.

This module provides the main interface that integrates all system components:
- Context management with compression and pruning
- Memory system with three-tier architecture
- Tool system for external integrations
- Storage system with hierarchical caching
- Performance monitoring and optimization
- Health checks and diagnostics
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

from .config import ConfigManager, ContextManagerConfig
from .models import ContextWindow, Message, SystemState, ContextPriority, Memory, ToolResult, MemoryType
from .memory_manager import MemoryManager
from .tool_manager import ToolManager
from ..storage.hierarchical_manager import StorageManager, StorageHierarchyConfig
from ..compression.compression_manager import CompressionManager
from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError, MemoryError, ToolError, StorageError


logger = get_logger(__name__)


class SystemStatus(Enum):
    """System status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    message_count: int = 0
    total_tokens: int = 0
    compression_count: int = 0
    pruning_count: int = 0
    memory_retrievals: int = 0
    tool_executions: int = 0
    storage_operations: int = 0
    errors: int = 0
    
    # Timing metrics
    average_message_time: float = 0.0
    average_compression_time: float = 0.0
    average_memory_time: float = 0.0
    average_tool_time: float = 0.0
    
    # Resource usage
    memory_usage: int = 0
    disk_usage: int = 0
    cpu_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class EnhancedContextManager:
    """
    Enhanced context manager with full system integration.
    
    This class provides a unified interface for:
    - Context window management with intelligent compression
    - Memory system with semantic search and retrieval
    - Tool system for external integrations
    - Storage system with hierarchical caching
    - Performance monitoring and optimization
    - Health checks and diagnostics
    """
    
    def __init__(self, config: Optional[ContextManagerConfig] = None, 
                 config_path: Optional[str] = None):
        """
        Initialize the enhanced context manager.
        
        Args:
            config: Optional configuration object
            config_path: Optional path to configuration file
        """
        # System state
        self._status = SystemStatus.INITIALIZING
        self._lock = threading.RLock()
        self._initialization_time = time.time()
        
        # Initialize configuration
        if config is None:
            self.config_manager = ConfigManager(config_path)
        else:
            self.config_manager = ConfigManager()
            self.config_manager.config = config
            
        self.config = self.config_manager.config
        
        # Initialize core components
        self.context_window = ContextWindow(max_tokens=self.config.max_tokens)
        self.system_state = SystemState(context_window=self.context_window)
        self.memory_manager = MemoryManager(self.config)
        self.tool_manager = ToolManager(self.config)
        
        # Initialize storage system
        storage_config = StorageHierarchyConfig(
            memory_size=self.config.memory_cache_size,
            memory_items=self.config.memory_cache_items,
            cache_size=self.config.cache_size,
            cache_items=self.config.cache_items,
            disk_path=self.config.storage_path,
            promotion_threshold=self.config.promotion_threshold
        )
        self.storage_manager = StorageManager(config=storage_config)
        
        # Initialize compression manager
        self.compression_manager = CompressionManager()
        
        # Context management state
        self.compression_history = deque(maxlen=100)
        self.pruning_stats = defaultdict(int)
        self.context_patterns = defaultdict(list)
        self.retrieval_cache = {}
        self.analysis_cache = {}
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics()
        self.message_times = deque(maxlen=1000)
        self.operation_times = defaultdict(lambda: deque(maxlen=100))
        
        # Health checks
        self.health_checks: List[HealthCheck] = []
        self.last_health_check = datetime.now()
        
        # Configuration for context management
        self.compression_threshold = 0.8  # Compress when 80% full
        self.pruning_threshold = 0.9      # Prune when 90% full
        self.min_context_size = 10        # Minimum messages to keep
        self.max_retrieval_results = 5   # Max memories to retrieve
        
        # Async task management
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = threading.Event()
        
        logger.info(f"EnhancedContextManager initialized with max_tokens={self.config.max_tokens}")
    
    async def initialize(self) -> None:
        """
        Initialize all system components.
        
        Raises:
            ContextManagerError: If initialization fails
        """
        try:
            logger.info("Initializing EnhancedContextManager...")
            
            # Initialize storage system
            await self.storage_manager.initialize()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Perform initial health checks
            await self._perform_initial_health_checks()
            
            # Update system status
            self._status = SystemStatus.READY
            logger.info("EnhancedContextManager initialized successfully")
            
        except Exception as e:
            self._status = SystemStatus.ERROR
            logger.error(f"Failed to initialize EnhancedContextManager: {e}")
            raise ContextManagerError(f"Initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all system resources."""
        try:
            logger.info("Cleaning up EnhancedContextManager...")
            
            # Update system status
            self._status = SystemStatus.STOPPING
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Clean up storage
            await self.storage_manager.cleanup()
            
            # Update system status
            self._status = SystemStatus.STOPPED
            logger.info("EnhancedContextManager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self._status = SystemStatus.ERROR
    
    async def add_message(self, role: str, content: str, **kwargs) -> bool:
        """
        Add a message to the context window with intelligent management.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message parameters
            
        Returns:
            True if message was added successfully
        """
        start_time = time.time()
        
        try:
            message = Message(role=role, content=content, **kwargs)
            
            # Check if we need to manage context before adding
            await self._manage_context_before_add(message)
            
            # Add the message
            success = self.context_window.add_message(message)
            
            if success:
                # Update context patterns and analysis
                self._update_context_patterns(message)
                await self._analyze_context_importance(message)
                
                # Store in memory system
                await self._store_message_in_memory(message)
                
                # Check if we need to manage context after adding
                await self._manage_context_after_add()
                
                # Update performance metrics
                self.performance_metrics.message_count += 1
                self.performance_metrics.total_tokens += message.token_count or 0
                
                logger.debug(f"Added message: {role} ({message.token_count} tokens)")
            
            # Update timing metrics
            elapsed_time = time.time() - start_time
            self.message_times.append(elapsed_time)
            self.performance_metrics.average_message_time = sum(self.message_times) / len(self.message_times)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            self.performance_metrics.errors += 1
            raise ContextManagerError(f"Failed to add message: {e}")
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool with proper error handling and monitoring.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If tool execution fails
        """
        start_time = time.time()
        
        try:
            # Execute tool
            result = await self.tool_manager.execute_tool_simple(tool_name, **kwargs)
            
            # Update performance metrics
            self.performance_metrics.tool_executions += 1
            
            # Store tool result in memory if applicable
            if isinstance(result, dict) and result.get('success'):
                await self._store_tool_result(tool_name, kwargs, result)
            
            # Update timing metrics
            elapsed_time = time.time() - start_time
            self.operation_times['tool_execution'].append(elapsed_time)
            self.performance_metrics.average_tool_time = (
                sum(self.operation_times['tool_execution']) / len(self.operation_times['tool_execution'])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            self.performance_metrics.errors += 1
            raise ToolError(f"Tool execution failed: {e}")
    
    async def search_memory(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search memory system with intelligent caching.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        start_time = time.time()
        
        try:
            # Check cache first - use a more specific cache key
            cache_key = f"memory_search:{hash(query + str(limit))}"
            if cache_key in self.retrieval_cache:
                cached_result = self.retrieval_cache[cache_key]
                if time.time() - cached_result['timestamp'] < 300:  # 5 minute cache
                    return cached_result['memories']
            
            # Perform search
            memories = self.memory_manager.search_memories(query, limit=limit)
            
            # Only cache non-empty results to avoid caching empty results from health checks
            if memories:
                self.retrieval_cache[cache_key] = {
                    'memories': memories,
                    'timestamp': time.time()
                }
            
            # Update performance metrics
            self.performance_metrics.memory_retrievals += 1
            
            # Update timing metrics
            elapsed_time = time.time() - start_time
            self.operation_times['memory_search'].append(elapsed_time)
            self.performance_metrics.average_memory_time = (
                sum(self.operation_times['memory_search']) / len(self.operation_times['memory_search'])
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            self.performance_metrics.errors += 1
            raise MemoryError(f"Memory search failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'status': self._status.value,
            'uptime': time.time() - self._initialization_time,
            'performance': self.performance_metrics.to_dict(),
            'context_summary': self.get_context_summary(),
            'storage_stats': self.storage_manager.get_stats(),
            'health_checks': [asdict(check) for check in self.health_checks[-10:]]
        }
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary containing context summary
        """
        return {
            "message_count": len(self.context_window.messages),
            "token_count": self.context_window.current_tokens,
            "max_tokens": self.context_window.max_tokens,
            "utilization": self.context_window.get_utilization(),
            "is_full": self.context_window.is_full(),
            "compression_history": len(self.compression_history),
            "memory_integrations": len([m for m in self.context_window.messages 
                                       if m.metadata.get("source") == "memory_retrieval"])
        }
    
    async def perform_health_check(self) -> List[HealthCheck]:
        """
        Perform comprehensive health checks.
        
        Returns:
            List of health check results
        """
        health_checks = []
        
        try:
            # Check system status
            health_checks.append(HealthCheck(
                name="system_status",
                status="healthy" if self._status == SystemStatus.READY else "degraded",
                message=f"System status: {self._status.value}"
            ))
            
            # Check context window
            context_utilization = self.context_window.get_utilization()
            health_checks.append(HealthCheck(
                name="context_window",
                status="healthy" if context_utilization < 0.8 else "warning",
                message=f"Context utilization: {context_utilization:.1%}",
                details={"utilization": context_utilization}
            ))
            
            # Check memory system
            memory_health = await self._check_memory_health()
            health_checks.append(memory_health)
            
            # Check storage system
            storage_health = await self._check_storage_health()
            health_checks.append(storage_health)
            
            # Check tool system
            tool_health = await self._check_tool_health()
            health_checks.append(tool_health)
            
            # Update health check history
            self.health_checks.extend(health_checks)
            self.last_health_check = datetime.now()
            
            return health_checks
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return [HealthCheck(
                name="health_check",
                status="error",
                message=f"Health check failed: {e}"
            )]
    
    async def optimize_system(self) -> Dict[str, Any]:
        """
        Optimize system performance.
        
        Returns:
            Dictionary containing optimization results
        """
        results = {
            'timestamp': datetime.now(),
            'actions_taken': [],
            'performance_improvements': {}
        }
        
        try:
            # Optimize context
            context_result = self.optimize_context()
            if context_result['actions_taken']:
                results['actions_taken'].extend(context_result['actions_taken'])
                results['performance_improvements']['context'] = context_result['optimized_state']
            
            # Optimize memory
            memory_result = await self._optimize_memory()
            if memory_result['actions_taken']:
                results['actions_taken'].extend(memory_result['actions_taken'])
                results['performance_improvements']['memory'] = memory_result['improvements']
            
            # Optimize storage
            storage_result = await self._optimize_storage()
            if storage_result['actions_taken']:
                results['actions_taken'].extend(storage_result['actions_taken'])
                results['performance_improvements']['storage'] = storage_result['improvements']
            
            logger.info(f"System optimization completed: {results['actions_taken']}")
            return results
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            raise ContextManagerError(f"Optimization failed: {e}")
    
    # Private Methods
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Health check task
        self._background_tasks['health_check'] = asyncio.create_task(
            self._health_check_loop()
        )
        
        # Performance monitoring task
        self._background_tasks['performance_monitor'] = asyncio.create_task(
            self._performance_monitoring_loop()
        )
        
        # Cache cleanup task
        self._background_tasks['cache_cleanup'] = asyncio.create_task(
            self._cache_cleanup_loop()
        )
    
    async def _stop_background_tasks(self) -> None:
        """Stop background maintenance tasks."""
        self._stop_event.set()
        
        for task_name, task in self._background_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._background_tasks.clear()
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                if not self._stop_event.is_set():
                    await self.perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                if not self._stop_event.is_set():
                    await self._update_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(600)  # Clean every 10 minutes
                if not self._stop_event.is_set():
                    await self._cleanup_caches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _manage_context_before_add(self, new_message: Message) -> None:
        """Manage context before adding a new message."""
        utilization = self.context_window.get_utilization()
        
        # If we're approaching limits, take action
        if utilization >= self.pruning_threshold:
            await self._prune_context()
        elif utilization >= self.compression_threshold:
            await self._compress_context()
        
        # Retrieve relevant memories if this is a user message
        if new_message.role == "user":
            await self._retrieve_relevant_context(new_message)
    
    async def _manage_context_after_add(self) -> None:
        """Manage context after adding a message."""
        utilization = self.context_window.get_utilization()
        
        # Check if we still need to manage context
        if utilization >= self.pruning_threshold:
            await self._prune_context()
        elif utilization >= self.compression_threshold:
            await self._compress_context()
    
    async def _compress_context(self) -> None:
        """Compress context to reduce token count."""
        if len(self.context_window.messages) < self.min_context_size:
            return
        
        start_time = time.time()
        
        original_count = len(self.context_window.messages)
        original_tokens = self.context_window.current_tokens
        
        # Group messages by role for better compression
        messages_by_role = defaultdict(list)
        for msg in self.context_window.messages:
            messages_by_role[msg.role].append(msg)
        
        compressed_messages = []
        
        # Compress each role group
        for role, messages in messages_by_role.items():
            if len(messages) <= 2:
                compressed_messages.extend(messages)
            else:
                # Keep the most recent message and compress older ones
                recent_message = messages[-1]
                older_messages = messages[:-1]
                
                # Create compressed summary
                compressed_summary = await self._create_message_summary(older_messages, role)
                compressed_messages.append(compressed_summary)
                compressed_messages.append(recent_message)
        
        # Replace context with compressed messages
        self.context_window.messages = compressed_messages
        self.context_window.current_tokens = sum(msg.token_count or 0 for msg in compressed_messages)
        
        # Record compression
        compression_record = {
            'timestamp': datetime.now(),
            'original_count': original_count,
            'original_tokens': original_tokens,
            'compressed_count': len(compressed_messages),
            'compressed_tokens': self.context_window.current_tokens,
            'compression_ratio': self.context_window.current_tokens / original_tokens
        }
        self.compression_history.append(compression_record)
        
        # Update performance metrics
        self.performance_metrics.compression_count += 1
        elapsed_time = time.time() - start_time
        self.operation_times['compression'].append(elapsed_time)
        self.performance_metrics.average_compression_time = (
            sum(self.operation_times['compression']) / len(self.operation_times['compression'])
        )
        
        logger.info(f"Context compressed: {original_count} -> {len(compressed_messages)} messages "
                   f"({original_tokens} -> {self.context_window.current_tokens} tokens)")
    
    async def _prune_context(self) -> None:
        """Prune less important messages from context."""
        if len(self.context_window.messages) <= self.min_context_size:
            return
        
        original_count = len(self.context_window.messages)
        
        # Sort messages by importance score
        messages_with_scores = []
        for msg in self.context_window.messages:
            score = await self._calculate_message_importance(msg)
            messages_with_scores.append((msg, score))
        
        # Sort by score (ascending) to remove least important first
        messages_with_scores.sort(key=lambda x: x[1])
        
        # Remove least important messages until we're under threshold
        target_count = max(self.min_context_size, int(len(self.context_window.messages) * 0.7))
        messages_to_remove = messages_with_scores[:len(messages_with_scores) - target_count]
        
        for msg, score in messages_to_remove:
            self.context_window.remove_message(msg.message_id)
            self.pruning_stats['removed_count'] += 1
            self.pruning_stats['removed_tokens'] += msg.token_count or 0
        
        # Update performance metrics
        self.performance_metrics.pruning_count += 1
        
        logger.info(f"Context pruned: {original_count} -> {len(self.context_window.messages)} messages")
    
    async def _retrieve_relevant_context(self, query_message: Message) -> None:
        """Retrieve relevant memories and add them to context."""
        try:
            # Search for relevant memories
            relevant_memories = await self.search_memory(
                query=query_message.content,
                limit=self.max_retrieval_results
            )
            
            if relevant_memories:
                # Create context message from memories
                context_content = await self._format_retrieved_context(relevant_memories)
                
                # Add as system message if not already present
                context_message = Message(
                    role="system",
                    content=context_content,
                    priority=ContextPriority.HIGH,
                    metadata={"source": "memory_retrieval", "memory_count": len(relevant_memories)}
                )
                
                # Check if similar context already exists
                if not await self._has_similar_context(context_message):
                    self.context_window.add_message(context_message)
                    logger.debug(f"Added retrieved context with {len(relevant_memories)} memories")
        
        except Exception as e:
            logger.warning(f"Failed to retrieve relevant context: {e}")
    
    async def _store_message_in_memory(self, message: Message) -> None:
        """Store message in memory system."""
        try:
            # Convert message to memory format
            # Map message role to memory type
            if message.role == "user":
                memory_type = MemoryType.SHORT_TERM
            elif message.role == "assistant":
                memory_type = MemoryType.SHORT_TERM
            elif message.role == "system":
                memory_type = MemoryType.MEDIUM_TERM
            else:
                memory_type = MemoryType.SHORT_TERM
                
            # Convert ContextPriority to importance score
            importance_map = {
                ContextPriority.CRITICAL: 0.9,
                ContextPriority.HIGH: 0.7,
                ContextPriority.MEDIUM: 0.5,
                ContextPriority.LOW: 0.3,
                ContextPriority.BACKGROUND: 0.1
            }
            importance = importance_map.get(message.priority, 0.5)
                
            self.memory_manager.add_memory(
                content=message.content,
                memory_type=memory_type,
                importance=importance,
                metadata={
                    "message_id": message.message_id,
                    "token_count": message.token_count,
                    "timestamp": message.timestamp
                }
            )
            
            # Clear retrieval cache when new memory is added to ensure fresh searches
            self.retrieval_cache.clear()
            
        except Exception as e:
            logger.warning(f"Failed to store message in memory: {e}")
    
    async def _store_tool_result(self, tool_name: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Store tool result in memory system."""
        try:
            memory_content = f"Tool execution: {tool_name}\nParameters: {json.dumps(params, indent=2)}\nResult: {json.dumps(result, indent=2)}"
            
            self.memory_manager.add_memory(
                content=memory_content,
                memory_type=MemoryType.SHORT_TERM,
                importance=0.5,  # Medium priority
                metadata={
                    "tool_name": tool_name,
                    "success": result.get('success', False),
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to store tool result in memory: {e}")
    
    async def _calculate_message_importance(self, message: Message) -> float:
        """Calculate importance score for a message."""
        score = 0.0
        
        # Priority score
        priority_scores = {
            ContextPriority.CRITICAL: 1.0,
            ContextPriority.HIGH: 0.8,
            ContextPriority.MEDIUM: 0.6,
            ContextPriority.LOW: 0.4,
            ContextPriority.BACKGROUND: 0.2
        }
        score += priority_scores.get(message.priority, 0.5)
        
        # Recency score
        age = time.time() - message.timestamp
        recency_score = max(0, 1 - age / (24 * 3600))
        score += recency_score * 0.3
        
        # Length score
        length_score = min(1.0, (message.token_count or 0) / 1000)
        score += length_score * 0.2
        
        # Role score
        role_scores = {"user": 0.8, "assistant": 0.7, "system": 0.5}
        score += role_scores.get(message.role, 0.5) * 0.2
        
        return min(score, 1.0)
    
    async def _create_message_summary(self, messages: List[Message], role: str) -> Message:
        """Create a summary message from a list of messages."""
        if not messages:
            return Message(role=role, content="[No content]")
        
        # Extract key information
        contents = [msg.content for msg in messages]
        
        # Simple summarization
        if len(contents) == 1:
            summary = contents[0]
        else:
            # Combine messages with timestamps
            summary_parts = []
            for msg in messages:
                timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
                summary_parts.append(f"[{timestamp}] {msg.content[:100]}...")
            summary = f"Summary of {len(messages)} {role} messages:\n" + "\n".join(summary_parts)
        
        return Message(
            role=role,
            content=summary,
            priority=ContextPriority.MEDIUM,
            metadata={"compressed": True, "original_count": len(messages)}
        )
    
    async def _format_retrieved_context(self, memories: List[Memory]) -> str:
        """Format retrieved memories as context."""
        if not memories:
            return ""
        
        context_parts = []
        for memory in memories:
            timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
            context_parts.append(f"[{timestamp}] {memory.content}")
        
        return f"Relevant context from memory:\n" + "\n".join(context_parts)
    
    async def _has_similar_context(self, new_message: Message) -> bool:
        """Check if similar context already exists."""
        for existing_msg in self.context_window.messages:
            if (existing_msg.role == new_message.role and 
                existing_msg.metadata.get("source") == "memory_retrieval"):
                similarity = await self._calculate_text_similarity(
                    existing_msg.content, new_message.content
                )
                if similarity > 0.7:
                    return True
        return False
    
    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _update_context_patterns(self, message: Message) -> None:
        """Update context pattern analysis."""
        pattern_key = f"{message.role}_{len(message.content.split())}_words"
        self.context_patterns[pattern_key].append(message.timestamp)
    
    async def _analyze_context_importance(self, message: Message) -> None:
        """Analyze and update context importance metrics."""
        if 'importance_analysis' not in self.analysis_cache:
            self.analysis_cache['importance_analysis'] = []
        
        importance_score = await self._calculate_message_importance(message)
        self.analysis_cache['importance_analysis'].append({
            'message_id': message.message_id,
            'importance': importance_score,
            'timestamp': datetime.now()
        })
    
    async def _perform_initial_health_checks(self) -> None:
        """Perform initial health checks."""
        await self.perform_health_check()
    
    async def _check_memory_health(self) -> HealthCheck:
        """Check memory system health."""
        try:
            # Try a simple memory operation
            self.memory_manager.add_memory(
                content="Health check test",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.3  # Low priority
            )
            
            # Try to retrieve it
            results = self.memory_manager.search_memories("health check", limit=1)
            
            if results:
                return HealthCheck(
                    name="memory_system",
                    status="healthy",
                    message="Memory system is functioning properly"
                )
            else:
                return HealthCheck(
                    name="memory_system",
                    status="warning",
                    message="Memory system may have retrieval issues"
                )
                
        except Exception as e:
            return HealthCheck(
                name="memory_system",
                status="error",
                message=f"Memory system error: {e}"
            )
    
    async def _check_storage_health(self) -> HealthCheck:
        """Check storage system health."""
        try:
            # Try a simple storage operation
            await self.storage_manager.set("health_check", "test_value")
            result = await self.storage_manager.get("health_check")
            
            if result == "test_value":
                return HealthCheck(
                    name="storage_system",
                    status="healthy",
                    message="Storage system is functioning properly"
                )
            else:
                return HealthCheck(
                    name="storage_system",
                    status="warning",
                    message="Storage system may have consistency issues"
                )
                
        except Exception as e:
            return HealthCheck(
                name="storage_system",
                status="error",
                message=f"Storage system error: {e}"
            )
    
    async def _check_tool_health(self) -> HealthCheck:
        """Check tool system health."""
        try:
            # List available tools
            tools = await self.tool_manager.list_tools()
            
            return HealthCheck(
                name="tool_system",
                status="healthy",
                message=f"Tool system is functioning properly with {len(tools)} tools available"
            )
                
        except Exception as e:
            return HealthCheck(
                name="tool_system",
                status="error",
                message=f"Tool system error: {e}"
            )
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Update resource usage
            import psutil
            process = psutil.Process()
            
            self.performance_metrics.memory_usage = process.memory_info().rss
            self.performance_metrics.cpu_usage = process.cpu_percent()
            
            # Update disk usage from storage
            storage_stats = self.storage_manager.get_stats()
            self.performance_metrics.disk_usage = storage_stats.get('total', {}).get('total_size', 0)
            
        except ImportError:
            # psutil not available, skip resource metrics
            pass
        except Exception as e:
            logger.warning(f"Failed to update performance metrics: {e}")
    
    async def _cleanup_caches(self) -> None:
        """Clean up expired cache entries."""
        try:
            current_time = time.time()
            
            # Clean retrieval cache
            expired_keys = [
                key for key, value in self.retrieval_cache.items()
                if current_time - value['timestamp'] > 3600  # 1 hour
            ]
            
            for key in expired_keys:
                del self.retrieval_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.warning(f"Failed to clean up caches: {e}")
    
    def optimize_context(self) -> Dict[str, Any]:
        """Optimize context by applying various strategies."""
        results = {
            'original_state': {
                'message_count': len(self.context_window.messages),
                'token_count': self.context_window.current_tokens,
                'utilization': self.context_window.get_utilization()
            },
            'actions_taken': []
        }
        
        # Apply compression if needed
        if self.context_window.get_utilization() > self.compression_threshold:
            # Note: This would need to be async in a real implementation
            results['actions_taken'].append('compression_needed')
        
        # Apply pruning if needed
        if self.context_window.get_utilization() > self.pruning_threshold:
            # Note: This would need to be async in a real implementation
            results['actions_taken'].append('pruning_needed')
        
        # Update results
        results['optimized_state'] = {
            'message_count': len(self.context_window.messages),
            'token_count': self.context_window.current_tokens,
            'utilization': self.context_window.get_utilization()
        }
        
        return results
    
    async def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory system."""
        # This would implement memory optimization strategies
        return {
            'actions_taken': [],
            'improvements': {}
        }
    
    async def _optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage system."""
        # This would implement storage optimization strategies
        return {
            'actions_taken': [],
            'improvements': {}
        }