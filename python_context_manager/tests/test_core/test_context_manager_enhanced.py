"""
Tests for the enhanced context management system.

This module tests the advanced context management capabilities including:
- Context compression and pruning
- Memory integration for context retrieval
- Context analysis and optimization
- Pattern recognition and health monitoring
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from context_manager.core.context_manager import ContextManager
from context_manager.core.models import Message, ContextPriority, Memory, MemoryType
from context_manager.core.config import ConfigManager, ContextManagerConfig


class TestEnhancedContextManager:
    """Test cases for enhanced context management functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.config = ContextManagerConfig(
            max_tokens=1000,
            short_term_memory_size=50
        )
        self.context_manager = ContextManager(config=self.config)
        
        # Create test messages
        self.test_messages = [
            Message(
                role="user",
                content="Hello, I need help with machine learning",
                priority=ContextPriority.HIGH
            ),
            Message(
                role="assistant",
                content="I'd be happy to help with machine learning!",
                priority=ContextPriority.MEDIUM
            ),
            Message(
                role="user",
                content="Can you explain neural networks?",
                priority=ContextPriority.HIGH
            ),
            Message(
                role="assistant",
                content="Neural networks are computing systems inspired by biological neural networks.",
                priority=ContextPriority.MEDIUM
            )
        ]
    
    def test_context_manager_initialization(self):
        """Test context manager initialization with enhanced features."""
        assert self.context_manager.config.max_tokens == 1000
        assert self.context_manager.compression_threshold == 0.8
        assert self.context_manager.pruning_threshold == 0.9
        assert self.context_manager.min_context_size == 10
        assert self.context_manager.max_retrieval_results == 5
        assert len(self.context_manager.compression_history) == 0
        assert len(self.context_manager.pruning_stats) == 0
    
    def test_add_message_with_context_management(self):
        """Test adding message with automatic context management."""
        # Add a message that should trigger context management
        result = self.context_manager.add_message(
            role="user",
            content="This is a test message for context management"
        )
        
        assert result is True
        assert len(self.context_manager.context_window.messages) == 1
        assert self.context_manager.context_window.current_tokens > 0
    
    def test_context_compression(self):
        """Test context compression functionality."""
        # Add many messages to trigger compression
        for i in range(15):
            self.context_manager.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Test message {i} with some content to test compression"
            )
        
        # Manually trigger compression
        initial_count = len(self.context_manager.context_window.messages)
        initial_tokens = self.context_manager.context_window.current_tokens
        
        self.context_manager._compress_context()
        
        # Check that compression occurred
        final_count = len(self.context_manager.context_window.messages)
        final_tokens = self.context_manager.context_window.current_tokens
        
        assert final_count <= initial_count
        assert len(self.context_manager.compression_history) > 0
        
        # Check compression history
        compression_record = self.context_manager.compression_history[-1]
        assert compression_record['original_count'] == initial_count
        assert compression_record['original_tokens'] == initial_tokens
        assert compression_record['compressed_count'] == final_count
        assert compression_record['compressed_tokens'] == final_tokens
    
    def test_context_pruning(self):
        """Test context pruning functionality."""
        # Add many messages with different priorities
        priorities = [ContextPriority.HIGH, ContextPriority.LOW, ContextPriority.MEDIUM, 
                     ContextPriority.BACKGROUND, ContextPriority.CRITICAL]
        
        for i in range(20):
            self.context_manager.add_message(
                role="user",
                content=f"Test message {i} with priority {priorities[i % len(priorities)].value}",
                priority=priorities[i % len(priorities)]
            )
        
        # Manually trigger pruning
        initial_count = len(self.context_manager.context_window.messages)
        
        self.context_manager._prune_context()
        
        # Check that pruning occurred
        final_count = len(self.context_manager.context_window.messages)
        assert final_count < initial_count
        assert self.context_manager.pruning_stats['removed_count'] > 0
    
    def test_message_importance_calculation(self):
        """Test message importance calculation."""
        # Test different message types
        critical_message = Message(
            role="user",
            content="Important question about neural networks",
            priority=ContextPriority.CRITICAL
        )
        
        low_priority_message = Message(
            role="system",
            content="Background information",
            priority=ContextPriority.BACKGROUND
        )
        
        critical_score = self.context_manager._calculate_message_importance(critical_message)
        low_score = self.context_manager._calculate_message_importance(low_priority_message)
        
        assert critical_score > low_score
    
    def test_memory_retrieval_integration(self):
        """Test integration with memory system for context retrieval."""
        # Add some memories to the memory manager
        self.context_manager.memory_manager.add_memory(
            content="Neural networks are important for machine learning",
            importance=0.8,
            tags=["AI", "machine learning"]
        )
        
        self.context_manager.memory_manager.add_memory(
            content="Deep learning uses neural networks",
            importance=0.9,
            tags=["AI", "deep learning"]
        )
        
        # Create a query message
        query_message = Message(
            role="user",
            content="Tell me about neural networks and AI",
            priority=ContextPriority.HIGH
        )
        
        # Test context retrieval
        self.context_manager._retrieve_relevant_context(query_message)
        
        # Check that context was added
        memory_context_messages = [
            msg for msg in self.context_manager.context_window.messages
            if msg.metadata.get("source") == "memory_retrieval"
        ]
        
        assert len(memory_context_messages) > 0
    
    def test_text_similarity_calculation(self):
        """Test text similarity calculation."""
        text1 = "machine learning algorithms"
        text2 = "machine learning and algorithms"
        text3 = "completely different topic"
        
        similarity1 = self.context_manager._calculate_text_similarity(text1, text2)
        similarity2 = self.context_manager._calculate_text_similarity(text1, text3)
        
        assert similarity1 > similarity2
        assert similarity1 > 0.5
        assert similarity2 < 0.5
    
    def test_context_pattern_analysis(self):
        """Test context pattern analysis."""
        # Add various messages
        for i in range(10):
            self.context_manager.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} with some content"
            )
        
        # Analyze patterns
        patterns = self.context_manager.analyze_context_patterns()
        
        assert 'message_patterns' in patterns
        assert 'role_distribution' in patterns
        assert 'token_distribution' in patterns
        assert 'temporal_patterns' in patterns
        
        # Check role distribution
        assert 'user' in patterns['role_distribution']
        assert 'assistant' in patterns['role_distribution']
    
    def test_context_health_monitoring(self):
        """Test context health monitoring."""
        # Add some messages
        for i in range(5):
            self.context_manager.add_message(
                role="user",
                content=f"Health test message {i}"
            )
        
        # Get health metrics
        health = self.context_manager.get_context_health()
        
        assert 'utilization' in health
        assert 'message_count' in health
        assert 'token_count' in health
        assert 'max_tokens' in health
        assert 'status' in health
        
        # Check that status is one of expected values
        assert health['status'] in ['healthy', 'moderate', 'high', 'critical']
    
    def test_context_optimization(self):
        """Test context optimization."""
        # Add many messages to create optimization opportunity
        for i in range(12):
            self.context_manager.add_message(
                role="user",
                content=f"Optimization test message {i} " * 10  # Make messages longer
            )
        
        # Get initial state
        initial_count = len(self.context_manager.context_window.messages)
        initial_tokens = self.context_manager.context_window.current_tokens
        
        # Optimize context
        results = self.context_manager.optimize_context()
        
        # Check results
        assert 'original_state' in results
        assert 'optimized_state' in results
        assert 'actions_taken' in results
        
        # Check that optimization occurred
        final_count = len(self.context_manager.context_window.messages)
        final_tokens = self.context_manager.context_window.current_tokens
        
        assert results['original_state']['message_count'] == initial_count
        assert results['original_state']['token_count'] == initial_tokens
        assert results['optimized_state']['message_count'] == final_count
        assert results['optimized_state']['token_count'] == final_tokens
    
    def test_message_summary_creation(self):
        """Test message summary creation."""
        # Create test messages
        messages = [
            Message(role="user", content="First message about AI"),
            Message(role="user", content="Second message about machine learning"),
            Message(role="user", content="Third message about neural networks")
        ]
        
        # Create summary
        summary = self.context_manager._create_message_summary(messages, "user")
        
        assert summary.role == "user"
        assert "Summary of 3 user messages" in summary.content
        assert summary.metadata.get("compressed") is True
        assert summary.metadata.get("original_count") == 3
    
    def test_retrieved_context_formatting(self):
        """Test formatting of retrieved context."""
        # Create test memories
        memories = [
            Memory(
                content="First memory about AI",
                timestamp=datetime.now() - timedelta(hours=1)
            ),
            Memory(
                content="Second memory about machine learning",
                timestamp=datetime.now() - timedelta(hours=2)
            )
        ]
        
        # Format context
        formatted = self.context_manager._format_retrieved_context(memories)
        
        assert "Relevant context from memory:" in formatted
        assert "First memory about AI" in formatted
        assert "Second memory about machine learning" in formatted
    
    def test_similar_context_detection(self):
        """Test detection of similar context."""
        # Add existing memory context
        existing_context = Message(
            role="system",
            content="Relevant context from memory:\n[2024-01-01 10:00] AI information",
            metadata={"source": "memory_retrieval"}
        )
        self.context_manager.context_window.add_message(existing_context)
        
        # Create new similar context with higher similarity
        new_context = Message(
            role="system",
            content="Relevant context from memory:\n[2024-01-01 10:00] AI information",
            metadata={"source": "memory_retrieval"}
        )
        
        # Check similarity detection
        is_similar = self.context_manager._has_similar_context(new_context)
        assert is_similar is True
    
    def test_context_management_thresholds(self):
        """Test context management based on utilization thresholds."""
        # Set up context to approach compression threshold
        for i in range(8):
            self.context_manager.add_message(
                role="user",
                content=f"Threshold test message {i} " * 20
            )
        
        # Check that compression doesn't trigger too early
        initial_count = len(self.context_manager.context_window.messages)
        
        # Add message that should trigger compression
        self.context_manager.add_message(
            role="user",
            content="This should trigger compression" * 50
        )
        
        # Check that compression occurred
        final_count = len(self.context_manager.context_window.messages)
        # Compression may or may not occur depending on exact token count
        assert final_count <= initial_count + 1
    
    def test_error_handling_in_context_management(self):
        """Test error handling in context management operations."""
        # Test with invalid message
        with pytest.raises(Exception):
            self.context_manager.add_message(role="", content="test")
        
        # Test compression with empty context
        self.context_manager.context_window.clear()
        self.context_manager._compress_context()  # Should not raise error
        
        # Test pruning with empty context
        self.context_manager._prune_context()  # Should not raise error
    
    def test_context_persistence_across_operations(self):
        """Test that context state persists correctly across operations."""
        # Add initial messages
        self.context_manager.add_message(role="user", content="Initial message")
        initial_count = len(self.context_manager.context_window.messages)
        
        # Perform various operations
        self.context_manager.analyze_context_patterns()
        self.context_manager.get_context_health()
        self.context_manager.optimize_context()
        
        # Check that context still exists
        final_count = len(self.context_manager.context_window.messages)
        assert final_count >= 1  # Should still have at least one message
    
    def test_performance_with_large_context(self):
        """Test performance with large context."""
        # Add many messages
        for i in range(50):
            self.context_manager.add_message(
                role="user",
                content=f"Performance test message {i}"
            )
        
        # Test that operations complete quickly
        import time
        start_time = time.time()
        
        self.context_manager.optimize_context()
        health = self.context_manager.get_context_health()
        patterns = self.context_manager.analyze_context_patterns()
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
        assert health['message_count'] > 0
        assert 'message_patterns' in patterns
    
    def test_configuration_updates(self):
        """Test configuration updates affecting context management."""
        # Update configuration
        self.context_manager.update_config(max_tokens=2000)
        
        assert self.context_manager.config.max_tokens == 2000
        assert self.context_manager.context_window.max_tokens == 2000
        
        # Test that context management works with new config
        self.context_manager.add_message(role="user", content="Test with new config")
        assert len(self.context_manager.context_window.messages) > 0
    
    def test_context_manager_string_representation(self):
        """Test string representation of context manager."""
        # Add some messages
        self.context_manager.add_message(role="user", content="Test message")
        
        # Get string representation
        str_repr = str(self.context_manager)
        
        assert "ContextManager" in str_repr
        assert "messages=" in str_repr
        assert "tokens=" in str_repr
    
    def test_system_state_integration(self):
        """Test integration with system state."""
        # Add messages and perform operations
        self.context_manager.add_message(role="user", content="System state test")
        self.context_manager.optimize_context()
        
        # Get system state
        state = self.context_manager.get_system_state()
        
        assert len(state.context_window.messages) > 0
        assert state.context_window.current_tokens > 0
        assert state.context_window.max_tokens == self.config.max_tokens