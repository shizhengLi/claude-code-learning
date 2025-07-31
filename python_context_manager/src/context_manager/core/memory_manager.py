"""
MemoryManager class for the three-tier memory architecture.

This module provides the memory management functionality with short-term,
medium-term, and long-term memory layers.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import re
from collections import defaultdict
from .config import ConfigManager, ContextManagerConfig
from .models import Memory, MemoryType, SystemState
from ..utils.logging import get_logger
from ..utils.error_handling import MemoryError


logger = get_logger(__name__)


class MemoryManager:
    """
    Memory manager implementing a three-tier memory architecture.
    
    This class manages:
    - Short-term memory: Recent messages and context
    - Medium-term memory: Important information from recent sessions
    - Long-term memory: Persistent knowledge and patterns
    """
    
    def __init__(self, config: Optional[ContextManagerConfig] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config
        else:
            self.config = config
            
        # Memory stores
        self.short_term_memory: List[Memory] = []
        self.medium_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        
        # Configuration
        self.short_term_limit = self.config.short_term_memory_size
        self.medium_term_limit = 1000  # Default medium term limit
        self.long_term_limit = 10000   # Default long term limit
        
        logger.info("MemoryManager initialized")
    
    def add_memory(self, content: str, memory_type: MemoryType = MemoryType.SHORT_TERM,
                   importance: float = 0.5, tags: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Memory:
        """
        Add a memory to the appropriate memory store.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for the memory
            metadata: Optional metadata
            
        Returns:
            Created memory object
        """
        try:
            memory = Memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Add to appropriate memory store
            if memory_type == MemoryType.SHORT_TERM:
                self.short_term_memory.append(memory)
                self._enforce_memory_limit(self.short_term_memory, self.short_term_limit)
            elif memory_type == MemoryType.MEDIUM_TERM:
                self.medium_term_memory.append(memory)
                self._enforce_memory_limit(self.medium_term_memory, self.medium_term_limit)
            elif memory_type == MemoryType.LONG_TERM:
                self.long_term_memory.append(memory)
                self._enforce_memory_limit(self.long_term_memory, self.long_term_limit)
            
            logger.debug(f"Added {memory_type.value} memory: {content[:50]}...")
            return memory
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise MemoryError(f"Failed to add memory: {e}")
    
    def get_memories(self, memory_type: Optional[MemoryType] = None, 
                    limit: Optional[int] = None, 
                    min_importance: float = 0.0) -> List[Memory]:
        """
        Retrieve memories from the specified memory store.
        
        Args:
            memory_type: Type of memory to retrieve (None for all)
            limit: Maximum number of memories to return
            min_importance: Minimum importance threshold
            
        Returns:
            List of memory objects
        """
        try:
            memories = []
            
            if memory_type is None or memory_type == MemoryType.SHORT_TERM:
                memories.extend(self.short_term_memory)
            if memory_type is None or memory_type == MemoryType.MEDIUM_TERM:
                memories.extend(self.medium_term_memory)
            if memory_type is None or memory_type == MemoryType.LONG_TERM:
                memories.extend(self.long_term_memory)
            
            # Filter by importance
            filtered_memories = [m for m in memories if m.importance >= min_importance]
            
            # Sort by importance and recency
            filtered_memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            
            # Apply limit
            if limit:
                filtered_memories = filtered_memories[:limit]
            
            return filtered_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise MemoryError(f"Failed to retrieve memories: {e}")
    
    def search_memories(self, query: str, memory_type: Optional[MemoryType] = None,
                       limit: int = 10, fuzzy_threshold: float = 0.6) -> List[Memory]:
        """
        Search for memories containing the query string with advanced search capabilities.
        
        Args:
            query: Search query
            memory_type: Type of memory to search
            limit: Maximum number of results
            fuzzy_threshold: Threshold for fuzzy matching (0.0 to 1.0)
            
        Returns:
            List of matching memory objects sorted by relevance
        """
        try:
            memories = self.get_memories(memory_type=memory_type)
            
            # Advanced search with multiple strategies
            matching_memories = []
            query_lower = query.lower()
            
            for memory in memories:
                relevance_score = self._calculate_relevance(memory, query_lower, fuzzy_threshold)
                if relevance_score > 0:
                    matching_memories.append((memory, relevance_score))
            
            # Sort by relevance score (combination of importance and search relevance)
            matching_memories.sort(key=lambda x: (
                x[1] * x[0].importance,  # Combined score
                x[0].timestamp  # Prefer more recent memories
            ), reverse=True)
            
            return [memory for memory, score in matching_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise MemoryError(f"Failed to search memories: {e}")
    
    def _calculate_relevance(self, memory: Memory, query_lower: str, fuzzy_threshold: float) -> float:
        """
        Calculate relevance score for a memory against a query.
        
        Args:
            memory: Memory object to score
            query_lower: Lowercase query string
            fuzzy_threshold: Threshold for fuzzy matching
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        
        # Exact content match
        if query_lower in memory.content.lower():
            score += 0.8
        
        # Word-based matching
        query_words = set(query_lower.split())
        content_words = set(memory.content.lower().split())
        word_overlap = len(query_words.intersection(content_words))
        if word_overlap > 0:
            score += min(word_overlap / len(query_words), 0.6)
        
        # Tag matching
        if memory.tags:
            matching_tags = sum(1 for tag in memory.tags if query_lower in tag.lower())
            score += min(matching_tags * 0.2, 0.4)
        
        # Fuzzy matching for approximate string similarity
        fuzzy_score = self._fuzzy_match(memory.content.lower(), query_lower)
        if fuzzy_score >= fuzzy_threshold:
            score += fuzzy_score * 0.3
        
        # Recency boost (newer memories get slight boost)
        days_old = (datetime.now() - memory.timestamp).days
        recency_boost = max(0, 1 - days_old / 365)  # Boost decreases over a year
        score += recency_boost * 0.1
        
        return min(score, 1.0)
    
    def _fuzzy_match(self, text: str, pattern: str) -> float:
        """
        Simple fuzzy string matching based on character overlap and sequence.
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            
        Returns:
            Fuzzy match score (0.0 to 1.0)
        """
        if not pattern or not text:
            return 0.0
        
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        
        # Character overlap ratio
        pattern_chars = set(pattern_lower)
        text_chars = set(text_lower)
        overlap = len(pattern_chars.intersection(text_chars))
        char_score = overlap / len(pattern_chars) if pattern_chars else 0.0
        
        # Sequence matching (bonus for consecutive characters)
        sequence_score = 0.0
        if len(pattern_lower) > 1:
            pattern_parts = pattern_lower.split()
            for part in pattern_parts:
                if part in text_lower:
                    sequence_score += len(part) / len(pattern_lower)
        
        # Combined score
        return (char_score * 0.7 + sequence_score * 0.3)
    
    def semantic_search(self, query: str, memory_type: Optional[MemoryType] = None,
                       limit: int = 10, similarity_threshold: float = 0.1) -> List[Memory]:
        """
        Perform semantic search based on concept similarity.
        
        Args:
            query: Search query
            memory_type: Type of memory to search
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of semantically similar memories
        """
        try:
            if not query or not query.strip():
                return []
                
            memories = self.get_memories(memory_type=memory_type)
            
            # Extract key concepts from query
            query_concepts = self._extract_concepts(query)
            
            if not query_concepts:
                return []
            
            # Score memories based on concept similarity
            scored_memories = []
            for memory in memories:
                memory_concepts = self._extract_concepts(memory.content)
                similarity = self._calculate_concept_similarity(query_concepts, memory_concepts)
                if similarity >= similarity_threshold:
                    scored_memories.append((memory, similarity))
            
            # Sort by semantic similarity
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            return [memory for memory, score in scored_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            raise MemoryError(f"Failed to perform semantic search: {e}")
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """
        Extract key concepts from text.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            Set of concept keywords
        """
        # Simple concept extraction using noun phrases and keywords
        concepts = set()
        
        # Extract words (simplified approach)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as',
                    'was', 'with', 'for', 'this', 'that', 'it', 'not', 'or', 'be',
                    'from', 'in', 'have', 'has', 'had', 'by', 'but', 'of', 'an'}
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                concepts.add(word)
        
        # Handle abbreviations like "AI"
        if "ai" in text.lower():
            concepts.add("ai")
        
        return concepts
    
    def _calculate_concept_similarity(self, concepts1: Set[str], concepts2: Set[str]) -> float:
        """
        Calculate similarity between two sets of concepts.
        
        Args:
            concepts1: First set of concepts
            concepts2: Second set of concepts
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not concepts1 or not concepts2:
            return 0.0
        
        intersection = len(concepts1.intersection(concepts2))
        union = len(concepts1.union(concepts2))
        
        return intersection / union if union > 0 else 0.0
    
    def search_by_tags(self, tags: List[str], memory_type: Optional[MemoryType] = None,
                      match_all: bool = False, limit: int = 10) -> List[Memory]:
        """
        Search memories by tags.
        
        Args:
            tags: List of tags to search for
            memory_type: Type of memory to search
            match_all: If True, requires all tags to match; if False, any tag match
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        try:
            memories = self.get_memories(memory_type=memory_type)
            matching_memories = []
            
            for memory in memories:
                if match_all:
                    if all(tag in memory.tags for tag in tags):
                        matching_memories.append(memory)
                else:
                    if any(tag in memory.tags for tag in tags):
                        matching_memories.append(memory)
            
            # Sort by importance and recency
            matching_memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            
            return matching_memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search by tags: {e}")
            raise MemoryError(f"Failed to search by tags: {e}")
    
    def search_by_time_range(self, start_time: datetime, end_time: datetime,
                           memory_type: Optional[MemoryType] = None,
                           limit: int = 10) -> List[Memory]:
        """
        Search memories within a specific time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            memory_type: Type of memory to search
            limit: Maximum number of results
            
        Returns:
            List of memories within the time range
        """
        try:
            memories = self.get_memories(memory_type=memory_type)
            matching_memories = []
            
            for memory in memories:
                if start_time <= memory.timestamp <= end_time:
                    matching_memories.append(memory)
            
            # Sort by timestamp (most recent first)
            matching_memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            return matching_memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search by time range: {e}")
            raise MemoryError(f"Failed to search by time range: {e}")
    
    def get_memory_clusters(self, cluster_size: int = 5) -> Dict[str, List[Memory]]:
        """
        Group memories into clusters based on content similarity.
        
        Args:
            cluster_size: Number of memories per cluster
            
        Returns:
            Dictionary mapping cluster labels to memory lists
        """
        try:
            all_memories = self.get_memories()
            clusters = defaultdict(list)
            
            # Simple clustering based on shared concepts
            for memory in all_memories:
                concepts = self._extract_concepts(memory.content)
                
                # Find the most important concept as cluster key
                if concepts:
                    main_concept = max(concepts, key=lambda c: len(c))
                    clusters[main_concept].append(memory)
            
            # Sort clusters by size and limit to cluster_size
            result = {}
            for cluster_label, cluster_memories in clusters.items():
                if len(cluster_memories) >= 2:  # Only include clusters with multiple memories
                    cluster_memories.sort(key=lambda m: m.importance, reverse=True)
                    result[cluster_label] = cluster_memories[:cluster_size]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get memory clusters: {e}")
            raise MemoryError(f"Failed to get memory clusters: {e}")
    
    def find_related_memories(self, target_memory: Memory, limit: int = 5) -> List[Memory]:
        """
        Find memories related to a target memory.
        
        Args:
            target_memory: Memory to find related memories for
            limit: Maximum number of related memories
            
        Returns:
            List of related memories
        """
        try:
            all_memories = self.get_memories()
            related_memories = []
            
            # Exclude the target memory itself
            other_memories = [m for m in all_memories if m != target_memory]
            
            for memory in other_memories:
                # Calculate relatedness score
                score = 0.0
                
                # Content similarity
                target_concepts = self._extract_concepts(target_memory.content)
                memory_concepts = self._extract_concepts(memory.content)
                content_similarity = self._calculate_concept_similarity(target_concepts, memory_concepts)
                score += content_similarity * 0.5
                
                # Tag similarity
                if target_memory.tags and memory.tags:
                    common_tags = len(set(target_memory.tags).intersection(set(memory.tags)))
                    tag_similarity = common_tags / len(set(target_memory.tags + memory.tags))
                    score += tag_similarity * 0.3
                
                # Time proximity (memories around the same time)
                time_diff = abs((target_memory.timestamp - memory.timestamp).total_seconds())
                time_proximity = max(0, 1 - time_diff / (7 * 24 * 3600))  # Within a week
                score += time_proximity * 0.2
                
                if score > 0.1:  # Minimum threshold
                    related_memories.append((memory, score))
            
            # Sort by relatedness score
            related_memories.sort(key=lambda x: x[1], reverse=True)
            
            return [memory for memory, score in related_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to find related memories: {e}")
            raise MemoryError(f"Failed to find related memories: {e}")
    
    def promote_memory(self, memory: Memory, target_type: MemoryType) -> bool:
        """
        Promote a memory to a different memory type.
        
        Args:
            memory: Memory to promote
            target_type: Target memory type
            
        Returns:
            True if promotion was successful
        """
        try:
            # Remove from current memory store
            if memory.memory_type == MemoryType.SHORT_TERM and memory in self.short_term_memory:
                self.short_term_memory.remove(memory)
            elif memory.memory_type == MemoryType.MEDIUM_TERM and memory in self.medium_term_memory:
                self.medium_term_memory.remove(memory)
            elif memory.memory_type == MemoryType.LONG_TERM and memory in self.long_term_memory:
                self.long_term_memory.remove(memory)
            else:
                return False
            
            # Update memory type
            memory.memory_type = target_type
            
            # Add to target memory store
            if target_type == MemoryType.SHORT_TERM:
                self.short_term_memory.append(memory)
            elif target_type == MemoryType.MEDIUM_TERM:
                self.medium_term_memory.append(memory)
            elif target_type == MemoryType.LONG_TERM:
                self.long_term_memory.append(memory)
            
            logger.info(f"Promoted memory to {target_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote memory: {e}")
            raise MemoryError(f"Failed to promote memory: {e}")
    
    def clear_memory(self, memory_type: Optional[MemoryType] = None) -> None:
        """
        Clear memories from the specified memory store.
        
        Args:
            memory_type: Type of memory to clear (None for all)
        """
        try:
            if memory_type is None or memory_type == MemoryType.SHORT_TERM:
                self.short_term_memory.clear()
            if memory_type is None or memory_type == MemoryType.MEDIUM_TERM:
                self.medium_term_memory.clear()
            if memory_type is None or memory_type == MemoryType.LONG_TERM:
                self.long_term_memory.clear()
                
            logger.info(f"Cleared {memory_type.value if memory_type else 'all'} memory")
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            raise MemoryError(f"Failed to clear memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        return {
            "short_term": {
                "count": len(self.short_term_memory),
                "limit": self.short_term_limit,
                "usage": len(self.short_term_memory) / self.short_term_limit
            },
            "medium_term": {
                "count": len(self.medium_term_memory),
                "limit": self.medium_term_limit,
                "usage": len(self.medium_term_memory) / self.medium_term_limit
            },
            "long_term": {
                "count": len(self.long_term_memory),
                "limit": self.long_term_limit,
                "usage": len(self.long_term_memory) / self.long_term_limit
            }
        }
    
    def _enforce_memory_limit(self, memory_list: List[Memory], limit: int) -> None:
        """
        Enforce memory limit by removing least important memories.
        
        Args:
            memory_list: List of memories to check
            limit: Maximum number of memories to keep
        """
        if len(memory_list) > limit:
            # Sort by importance and remove least important
            memory_list.sort(key=lambda m: (m.importance, m.timestamp))
            while len(memory_list) > limit:
                memory_list.pop(0)