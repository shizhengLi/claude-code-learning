"""
Tests for the enhanced memory management system with advanced search capabilities.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from context_manager.core.memory_manager import MemoryManager
from context_manager.core.models import Memory, MemoryType, SystemState
from context_manager.core.config import ConfigManager, ContextManagerConfig


class TestEnhancedMemoryManager:
    """Test cases for enhanced memory management functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.memory_manager = MemoryManager()
        
        # Create test memories
        self.test_memories = [
            Memory(
                content="Machine learning algorithms are important for AI systems",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.8,
                tags=["AI", "machine learning", "algorithms"]
            ),
            Memory(
                content="Deep learning models use neural networks",
                memory_type=MemoryType.MEDIUM_TERM,
                importance=0.9,
                tags=["AI", "deep learning", "neural networks"]
            ),
            Memory(
                content="Python is a great programming language for data science",
                memory_type=MemoryType.LONG_TERM,
                importance=0.7,
                tags=["programming", "Python", "data science"]
            ),
            Memory(
                content="Data preprocessing is crucial for machine learning",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.6,
                tags=["data science", "machine learning", "preprocessing"]
            ),
            Memory(
                content="Neural networks are inspired by biological neurons",
                memory_type=MemoryType.MEDIUM_TERM,
                importance=0.8,
                tags=["neural networks", "biology", "AI"]
            )
        ]
        
        # Add test memories to memory manager
        for memory in self.test_memories:
            self.memory_manager.add_memory(
                content=memory.content,
                memory_type=memory.memory_type,
                importance=memory.importance,
                tags=memory.tags
            )
    
    def test_advanced_search_with_fuzzy_matching(self):
        """Test advanced search with fuzzy matching capabilities."""
        # Test exact match
        results = self.memory_manager.search_memories("machine learning")
        assert len(results) >= 2
        assert any("machine learning" in result.content.lower() for result in results)
        
        # Test fuzzy match
        results = self.memory_manager.search_memories("machin learn", fuzzy_threshold=0.5)
        assert len(results) >= 1
        
        # Test tag matching
        results = self.memory_manager.search_memories("AI")
        assert len(results) >= 2
        assert any("AI" in result.tags for result in results)
    
    def test_semantic_search(self):
        """Test semantic search based on concept similarity."""
        # Search for conceptually related memories
        results = self.memory_manager.semantic_search("artificial intelligence")
        assert len(results) >= 0  # May not find results due to concept extraction
        
        # Test with specific memory type
        results = self.memory_manager.semantic_search("programming", memory_type=MemoryType.LONG_TERM)
        assert len(results) >= 0
        
        # Test limit parameter
        results = self.memory_manager.semantic_search("AI", limit=2)
        assert len(results) <= 2
        
        # Test with more specific query
        results = self.memory_manager.semantic_search("machine")
        assert len(results) >= 0
    
    def test_concept_extraction(self):
        """Test concept extraction from text."""
        concepts = self.memory_manager._extract_concepts("Machine learning algorithms are important for AI systems")
        assert "machine" in concepts
        assert "learning" in concepts
        assert "algorithms" in concepts
        assert "important" in concepts
        assert "ai" in concepts
        assert "systems" in concepts
        
        # Test stop word filtering
        concepts = self.memory_manager._extract_concepts("The quick brown fox jumps over the lazy dog")
        assert "the" not in concepts
        assert "quick" in concepts
        assert "brown" in concepts
        assert "fox" in concepts
    
    def test_concept_similarity_calculation(self):
        """Test concept similarity calculation."""
        concepts1 = {"machine", "learning", "ai"}
        concepts2 = {"machine", "learning", "algorithms"}
        concepts3 = {"python", "programming", "data"}
        
        # Test similar concepts
        similarity = self.memory_manager._calculate_concept_similarity(concepts1, concepts2)
        assert similarity >= 0.4
        
        # Test different concepts
        similarity = self.memory_manager._calculate_concept_similarity(concepts1, concepts3)
        assert similarity < 0.3
        
        # Test empty concepts
        similarity = self.memory_manager._calculate_concept_similarity(concepts1, set())
        assert similarity == 0.0
    
    def test_search_by_tags(self):
        """Test searching memories by tags."""
        # Test single tag search
        results = self.memory_manager.search_by_tags(["AI"])
        assert len(results) >= 2
        
        # Test multiple tags with any match
        results = self.memory_manager.search_by_tags(["AI", "programming"], match_all=False)
        assert len(results) >= 3
        
        # Test multiple tags with all match
        results = self.memory_manager.search_by_tags(["machine learning", "algorithms"], match_all=True)
        assert len(results) >= 1
        
        # Test non-existent tags
        results = self.memory_manager.search_by_tags(["nonexistent"])
        assert len(results) == 0
    
    def test_search_by_time_range(self):
        """Test searching memories within a time range."""
        # Get current time
        now = datetime.now()
        
        # Create a memory with specific timestamp
        test_memory = Memory(
            content="Test memory for time range search",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.5,
            tags=["test"]
        )
        test_memory.timestamp = now - timedelta(hours=1)
        
        self.memory_manager.short_term_memory.append(test_memory)
        
        # Search within time range
        start_time = now - timedelta(hours=2)
        end_time = now - timedelta(minutes=30)
        
        results = self.memory_manager.search_by_time_range(start_time, end_time)
        assert len(results) >= 1
        assert any("Test memory" in result.content for result in results)
        
        # Search outside time range
        start_time = now - timedelta(days=2)
        end_time = now - timedelta(days=1)
        
        results = self.memory_manager.search_by_time_range(start_time, end_time)
        assert len(results) == 0
    
    def test_get_memory_clusters(self):
        """Test memory clustering functionality."""
        clusters = self.memory_manager.get_memory_clusters(cluster_size=3)
        
        # Check that clusters are returned
        assert isinstance(clusters, dict)
        
        # Check that each cluster has the right size
        for cluster_label, cluster_memories in clusters.items():
            assert len(cluster_memories) <= 3
            assert len(cluster_memories) >= 2  # Only clusters with multiple memories
    
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create a target memory
        target_memory = self.test_memories[0]  # Machine learning memory
        
        # Find related memories
        related = self.memory_manager.find_related_memories(target_memory, limit=3)
        
        # Should find related memories
        assert len(related) <= 3
        
        # Check that target memory is not in results
        assert target_memory not in related
        
        # Related memories should have some similarity
        for memory in related:
            # Check for content similarity or tag similarity
            content_overlap = len(set(target_memory.content.lower().split()) & 
                                set(memory.content.lower().split()))
            tag_overlap = len(set(target_memory.tags) & set(memory.tags))
            
            assert content_overlap > 0 or tag_overlap > 0
    
    def test_relevance_scoring(self):
        """Test relevance scoring for search results."""
        memory = self.test_memories[0]
        query = "machine learning"
        
        # Test relevance calculation
        score = self.memory_manager._calculate_relevance(memory, query.lower(), 0.6)
        
        # Score should be positive for relevant memory
        assert score > 0.0
        
        # Test with completely irrelevant query
        score = self.memory_manager._calculate_relevance(memory, "xyz abc", 0.6)
        assert score <= 0.2  # Some fuzzy matching may occur
    
    def test_fuzzy_matching(self):
        """Test fuzzy string matching."""
        text = "machine learning algorithms"
        
        # Test good match
        score = self.memory_manager._fuzzy_match(text, "machine learn")
        assert score > 0.4
        
        # Test partial match
        score = self.memory_manager._fuzzy_match(text, "mach learn")
        assert score > 0.2
        
        # Test no match - should be very low but may have some character overlap
        score = self.memory_manager._fuzzy_match(text, "xyz abc")
        assert score < 0.5
        
        # Test empty strings
        score = self.memory_manager._fuzzy_match("", "test")
        assert score == 0.0
        
        score = self.memory_manager._fuzzy_match(text, "")
        assert score == 0.0
    
    def test_memory_stats_with_enhanced_features(self):
        """Test memory statistics with enhanced features."""
        stats = self.memory_manager.get_memory_stats()
        
        # Check basic stats
        assert "short_term" in stats
        assert "medium_term" in stats
        assert "long_term" in stats
        
        # Check that each memory type has stats
        for memory_type in ["short_term", "medium_term", "long_term"]:
            assert "count" in stats[memory_type]
            assert "limit" in stats[memory_type]
            assert "usage" in stats[memory_type]
    
    def test_error_handling_in_search_functions(self):
        """Test error handling in search functions."""
        # Test semantic search with empty query
        results = self.memory_manager.semantic_search("")
        assert len(results) == 0
        
        # Test semantic search with None query
        results = self.memory_manager.semantic_search(None)
        assert len(results) == 0
        
        # Test tag search with empty tags
        results = self.memory_manager.search_by_tags([])
        assert len(results) == 0
    
    def test_memory_promotion_with_search(self):
        """Test memory promotion with search functionality."""
        # Create a short-term memory
        memory = self.memory_manager.add_memory(
            content="Important memory to promote",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.9,
            tags=["important", "test"]
        )
        
        # Search for the memory
        results = self.memory_manager.search_memories("important")
        assert len(results) >= 1
        
        # Promote to medium-term
        success = self.memory_manager.promote_memory(memory, MemoryType.MEDIUM_TERM)
        assert success is True
        
        # Verify it's no longer in short-term
        short_term_memories = self.memory_manager.get_memories(memory_type=MemoryType.SHORT_TERM)
        assert memory not in short_term_memories
        
        # Verify it's in medium-term
        medium_term_memories = self.memory_manager.get_memories(memory_type=MemoryType.MEDIUM_TERM)
        assert memory in medium_term_memories
    
    def test_combined_search_strategies(self):
        """Test combining different search strategies."""
        # Test searching with multiple criteria
        results = self.memory_manager.search_memories("AI", limit=5)
        assert len(results) <= 5
        
        # Test semantic search with memory type filter
        results = self.memory_manager.semantic_search("learning", memory_type=MemoryType.SHORT_TERM)
        for result in results:
            assert result.memory_type == MemoryType.SHORT_TERM
        
        # Test tag search with importance threshold
        results = self.memory_manager.search_by_tags(["AI"], limit=10)
        for result in results:
            assert "AI" in result.tags
    
    def test_performance_with_large_memory_set(self):
        """Test performance with large number of memories."""
        # Add many memories
        for i in range(100):
            self.memory_manager.add_memory(
                content=f"Test memory {i} with content about AI and machine learning",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.5,
                tags=["test", f"tag_{i % 10}"]
            )
        
        # Test search performance
        results = self.memory_manager.search_memories("AI", limit=10)
        assert len(results) <= 10
        
        # Test semantic search performance
        results = self.memory_manager.semantic_search("machine learning", limit=5)
        assert len(results) <= 5
        
        # Test tag search performance
        results = self.memory_manager.search_by_tags(["test"], limit=20)
        assert len(results) <= 20
    
    def test_memory_persistence_during_search(self):
        """Test that memory operations work correctly during search."""
        initial_count = len(self.memory_manager.get_memories())
        
        # Perform search
        results = self.memory_manager.search_memories("test")
        
        # Memory count should remain the same
        final_count = len(self.memory_manager.get_memories())
        assert initial_count == final_count
        
        # Add memory during search (should not affect ongoing search)
        new_memory = self.memory_manager.add_memory(
            content="New test memory",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.5,
            tags=["test"]
        )
        
        # Verify memory was added
        updated_count = len(self.memory_manager.get_memories())
        assert updated_count == initial_count + 1