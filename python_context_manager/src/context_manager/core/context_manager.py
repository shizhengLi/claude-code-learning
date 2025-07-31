"""
Main ContextManager class for the context management system.

This module provides the primary interface for managing context, memory,
and tool operations in the system.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import re
from collections import defaultdict, deque
from .config import ConfigManager, ContextManagerConfig
from .models import ContextWindow, Message, SystemState, ContextPriority, Memory
from .memory_manager import MemoryManager
from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError, MemoryError


logger = get_logger(__name__)


class ContextManager:
    """
    Main context manager class that coordinates all system components.
    
    This class provides the primary interface for:
    - Context window management with compression and pruning
    - Memory integration for context retrieval
    - Context analysis and optimization
    - Tool execution coordination
    - Configuration management
    """
    
    def __init__(self, config: Optional[ContextManagerConfig] = None, 
                 config_path: Optional[str] = None):
        """
        Initialize the context manager.
        
        Args:
            config: Optional configuration object
            config_path: Optional path to configuration file
        """
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
        
        # Context management state
        self.compression_history = deque(maxlen=100)
        self.pruning_stats = defaultdict(int)
        self.context_patterns = defaultdict(list)
        self.retrieval_cache = {}
        self.analysis_cache = {}
        
        # Configuration for context management
        self.compression_threshold = 0.8  # Compress when 80% full
        self.pruning_threshold = 0.9      # Prune when 90% full
        self.min_context_size = 10        # Minimum messages to keep
        self.max_retrieval_results = 5   # Max memories to retrieve
        
        logger.info(f"ContextManager initialized with max_tokens={self.config.max_tokens}")
    
    def add_message(self, role: str, content: str, **kwargs) -> bool:
        """
        Add a message to the context window with automatic context management.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message parameters
            
        Returns:
            True if message was added successfully
        """
        try:
            message = Message(role=role, content=content, **kwargs)
            
            # Check if we need to manage context before adding
            self._manage_context_before_add(message)
            
            # Add the message
            success = self.context_window.add_message(message)
            
            if success:
                # Update context patterns and analysis
                self._update_context_patterns(message)
                self._analyze_context_importance(message)
                
                # Check if we need to manage context after adding
                self._manage_context_after_add()
                
                logger.debug(f"Added message: {role} ({message.token_count} tokens)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise ContextManagerError(f"Failed to add message: {e}")
    
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
            "is_full": self.context_window.is_full()
        }
    
    def clear_context(self) -> None:
        """Clear all messages from the context window."""
        self.context_window.messages.clear()
        self.context_window.current_tokens = 0
        logger.info("Context cleared")
    
    def get_system_state(self) -> SystemState:
        """Get the current system state."""
        return self.system_state
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config_manager.update_config(**kwargs)
        self.config = self.config_manager.config
        
        # Update context window if max_tokens changed
        if 'max_tokens' in kwargs:
            self.context_window.max_tokens = kwargs['max_tokens']
            
        logger.info(f"Configuration updated: {kwargs}")
    
    def __str__(self) -> str:
        """String representation of the context manager."""
        return f"ContextManager(messages={len(self.context_window.messages)}, tokens={self.context_window.current_tokens}/{self.context_window.max_tokens})"
    
    # Context Management Methods
    
    def _manage_context_before_add(self, new_message: Message) -> None:
        """
        Manage context before adding a new message.
        
        Args:
            new_message: Message that will be added
        """
        utilization = self.context_window.get_utilization()
        
        # If we're approaching limits, take action
        if utilization >= self.pruning_threshold:
            self._prune_context()
        elif utilization >= self.compression_threshold:
            self._compress_context()
        
        # Retrieve relevant memories if this is a user message
        if new_message.role == "user":
            self._retrieve_relevant_context(new_message)
    
    def _manage_context_after_add(self) -> None:
        """Manage context after adding a message."""
        utilization = self.context_window.get_utilization()
        
        # Check if we still need to manage context
        if utilization >= self.pruning_threshold:
            self._prune_context()
        elif utilization >= self.compression_threshold:
            self._compress_context()
    
    def _compress_context(self) -> None:
        """Compress context to reduce token count."""
        if len(self.context_window.messages) < self.min_context_size:
            return
            
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
                compressed_summary = self._create_message_summary(older_messages, role)
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
        
        logger.info(f"Context compressed: {original_count} -> {len(compressed_messages)} messages "
                   f"({original_tokens} -> {self.context_window.current_tokens} tokens)")
    
    def _prune_context(self) -> None:
        """Prune less important messages from context."""
        if len(self.context_window.messages) <= self.min_context_size:
            return
            
        original_count = len(self.context_window.messages)
        
        # Sort messages by importance score
        messages_with_scores = []
        for msg in self.context_window.messages:
            score = self._calculate_message_importance(msg)
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
        
        logger.info(f"Context pruned: {original_count} -> {len(self.context_window.messages)} messages")
    
    def _retrieve_relevant_context(self, query_message: Message) -> None:
        """
        Retrieve relevant memories and add them to context.
        
        Args:
            query_message: Message to use as query for retrieval
        """
        try:
            # Search for relevant memories
            relevant_memories = self.memory_manager.search_memories(
                query=query_message.content,
                limit=self.max_retrieval_results
            )
            
            if relevant_memories:
                # Create context message from memories
                context_content = self._format_retrieved_context(relevant_memories)
                
                # Add as system message if not already present
                context_message = Message(
                    role="system",
                    content=context_content,
                    priority=ContextPriority.HIGH,
                    metadata={"source": "memory_retrieval", "memory_count": len(relevant_memories)}
                )
                
                # Check if similar context already exists
                if not self._has_similar_context(context_message):
                    self.context_window.add_message(context_message)
                    logger.debug(f"Added retrieved context with {len(relevant_memories)} memories")
        
        except Exception as e:
            logger.warning(f"Failed to retrieve relevant context: {e}")
    
    def _calculate_message_importance(self, message: Message) -> float:
        """
        Calculate importance score for a message.
        
        Args:
            message: Message to score
            
        Returns:
            Importance score (0.0 to 1.0)
        """
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
        
        # Recency score (more recent = higher score)
        age = datetime.now().timestamp() - message.timestamp
        recency_score = max(0, 1 - age / (24 * 3600))  # 24 hour decay
        score += recency_score * 0.3
        
        # Length score (longer messages might be more important)
        length_score = min(1.0, (message.token_count or 0) / 1000)
        score += length_score * 0.2
        
        # Role score (user and assistant messages might be more important)
        role_scores = {"user": 0.8, "assistant": 0.7, "system": 0.5}
        score += role_scores.get(message.role, 0.5) * 0.2
        
        return min(score, 1.0)
    
    def _create_message_summary(self, messages: List[Message], role: str) -> Message:
        """
        Create a summary message from a list of messages.
        
        Args:
            messages: Messages to summarize
            role: Role of the messages
            
        Returns:
            Summary message
        """
        if not messages:
            return Message(role=role, content="[No content]")
        
        # Extract key information
        contents = [msg.content for msg in messages]
        
        # Simple summarization - in practice, you might use an LLM
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
    
    def _format_retrieved_context(self, memories: List[Memory]) -> str:
        """
        Format retrieved memories as context.
        
        Args:
            memories: List of memories to format
            
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
        
        context_parts = []
        for memory in memories:
            timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
            context_parts.append(f"[{timestamp}] {memory.content}")
        
        return f"Relevant context from memory:\n" + "\n".join(context_parts)
    
    def _has_similar_context(self, new_message: Message) -> bool:
        """
        Check if similar context already exists.
        
        Args:
            new_message: Message to check
            
        Returns:
            True if similar context exists
        """
        for existing_msg in self.context_window.messages:
            if (existing_msg.role == new_message.role and 
                existing_msg.metadata.get("source") == "memory_retrieval"):
                # Simple similarity check
                similarity = self._calculate_text_similarity(
                    existing_msg.content, new_message.content
                )
                if similarity > 0.7:
                    return True
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _update_context_patterns(self, message: Message) -> None:
        """
        Update context pattern analysis.
        
        Args:
            message: New message to analyze
        """
        # Extract patterns based on role and content
        pattern_key = f"{message.role}_{len(message.content.split())}_words"
        self.context_patterns[pattern_key].append(message.timestamp)
    
    def _analyze_context_importance(self, message: Message) -> None:
        """
        Analyze and update context importance metrics.
        
        Args:
            message: Message to analyze
        """
        # Update analysis cache
        if 'importance_analysis' not in self.analysis_cache:
            self.analysis_cache['importance_analysis'] = []
        
        importance_score = self._calculate_message_importance(message)
        self.analysis_cache['importance_analysis'].append({
            'message_id': message.message_id,
            'importance': importance_score,
            'timestamp': datetime.now()
        })
    
    # Context Analysis Methods
    
    def analyze_context_patterns(self) -> Dict[str, Any]:
        """
        Analyze context patterns and return insights.
        
        Returns:
            Dictionary containing pattern analysis
        """
        analysis = {
            'message_patterns': {},
            'role_distribution': {},
            'token_distribution': {},
            'temporal_patterns': {}
        }
        
        # Analyze message patterns
        for pattern, timestamps in self.context_patterns.items():
            analysis['message_patterns'][pattern] = {
                'count': len(timestamps),
                'frequency': len(timestamps) / max(1, len(self.context_window.messages))
            }
        
        # Analyze role distribution
        role_counts = defaultdict(int)
        role_tokens = defaultdict(int)
        for msg in self.context_window.messages:
            role_counts[msg.role] += 1
            role_tokens[msg.role] += msg.token_count or 0
        
        analysis['role_distribution'] = dict(role_counts)
        analysis['token_distribution'] = dict(role_tokens)
        
        # Analyze temporal patterns
        if self.context_window.messages:
            timestamps = [msg.timestamp for msg in self.context_window.messages]
            analysis['temporal_patterns'] = {
                'span': max(timestamps) - min(timestamps),
                'average_interval': (max(timestamps) - min(timestamps)) / max(1, len(timestamps) - 1)
            }
        
        return analysis
    
    def get_context_health(self) -> Dict[str, Any]:
        """
        Get context health metrics.
        
        Returns:
            Dictionary containing health metrics
        """
        utilization = self.context_window.get_utilization()
        
        health = {
            'utilization': utilization,
            'message_count': len(self.context_window.messages),
            'token_count': self.context_window.current_tokens,
            'max_tokens': self.context_window.max_tokens,
            'compression_count': len(self.compression_history),
            'pruning_stats': dict(self.pruning_stats),
            'memory_retrieval_count': len([m for m in self.context_window.messages 
                                          if m.metadata.get("source") == "memory_retrieval"])
        }
        
        # Health status
        if utilization < 0.5:
            health['status'] = 'healthy'
        elif utilization < 0.8:
            health['status'] = 'moderate'
        elif utilization < 0.95:
            health['status'] = 'high'
        else:
            health['status'] = 'critical'
        
        return health
    
    def get_compression_history(self) -> List[Dict[str, Any]]:
        """
        Get compression history.
        
        Returns:
            List of compression records
        """
        return list(self.compression_history)
    
    def optimize_context(self) -> Dict[str, Any]:
        """
        Optimize context by applying various strategies.
        
        Returns:
            Dictionary containing optimization results
        """
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
            self._compress_context()
            results['actions_taken'].append('compression')
        
        # Apply pruning if needed
        if self.context_window.get_utilization() > self.pruning_threshold:
            self._prune_context()
            results['actions_taken'].append('pruning')
        
        # Update results
        results['optimized_state'] = {
            'message_count': len(self.context_window.messages),
            'token_count': self.context_window.current_tokens,
            'utilization': self.context_window.get_utilization()
        }
        
        return results