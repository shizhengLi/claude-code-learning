# Enhanced Memory Management System

## Overview

The enhanced memory management system extends the basic three-tier memory architecture with advanced search and retrieval capabilities. This system provides intelligent memory access through multiple search strategies, semantic analysis, and relevance scoring.

## Features

### 1. Advanced Search Capabilities

#### Fuzzy Matching
- Character overlap analysis
- Sequence matching bonus
- Configurable threshold parameters
- Handles misspellings and approximate matches

#### Semantic Search
- Concept extraction from text
- Stop word filtering
- Concept similarity calculation
- Threshold-based filtering

#### Tag-Based Search
- Single and multiple tag search
- Configurable match requirements (any/all)
- Combined with importance scoring

#### Time-Based Search
- Date range filtering
- Temporal memory retrieval
- Chronological ordering

### 2. Memory Analysis

#### Relevance Scoring
- Multi-factor scoring system:
  - Exact content matches (0.8 weight)
  - Word-based overlap (0.6 weight)
  - Tag matching (0.2-0.4 weight)
  - Fuzzy matching (0.3 weight)
  - Recency boost (0.1 weight)

#### Concept Extraction
- Stop word filtering
- Minimum word length filtering
- Abbreviation handling (e.g., "AI")
- Set-based concept representation

#### Memory Clustering
- Content similarity grouping
- Concept-based clustering
- Configurable cluster sizes
- Importance-based sorting

### 3. Related Memory Discovery

#### Memory Relationships
- Content similarity analysis
- Tag overlap detection
- Time proximity calculation
- Multi-factor scoring

#### Relationship Scoring
- Content similarity (50% weight)
- Tag similarity (30% weight)
- Time proximity (20% weight)
- Configurable thresholds

## API Reference

### Core Methods

#### `search_memories(query, memory_type=None, limit=10, fuzzy_threshold=0.6)`
Advanced text search with fuzzy matching and relevance scoring.

**Parameters:**
- `query`: Search string
- `memory_type`: Optional memory type filter
- `limit`: Maximum results
- `fuzzy_threshold`: Fuzzy matching threshold

**Returns:** List of Memory objects sorted by relevance

#### `semantic_search(query, memory_type=None, limit=10, similarity_threshold=0.1)`
Concept-based semantic search.

**Parameters:**
- `query`: Search string
- `memory_type`: Optional memory type filter
- `limit`: Maximum results
- `similarity_threshold`: Minimum similarity score

**Returns:** List of semantically similar Memory objects

#### `search_by_tags(tags, memory_type=None, match_all=False, limit=10)`
Tag-based memory search.

**Parameters:**
- `tags`: List of tags to search
- `memory_type`: Optional memory type filter
- `match_all`: True for all tags, False for any tag
- `limit`: Maximum results

**Returns:** List of matching Memory objects

#### `search_by_time_range(start_time, end_time, memory_type=None, limit=10)`
Time-based memory search.

**Parameters:**
- `start_time`: Range start datetime
- `end_time`: Range end datetime
- `memory_type`: Optional memory type filter
- `limit`: Maximum results

**Returns:** List of Memory objects within time range

#### `get_memory_clusters(cluster_size=5)`
Group memories into content-based clusters.

**Parameters:**
- `cluster_size`: Maximum memories per cluster

**Returns:** Dictionary mapping cluster labels to memory lists

#### `find_related_memories(target_memory, limit=5)`
Find memories related to a target memory.

**Parameters:**
- `target_memory`: Memory to find relations for
- `limit`: Maximum related memories

**Returns:** List of related Memory objects

### Utility Methods

#### `_calculate_relevance(memory, query, fuzzy_threshold)`
Calculate relevance score for memory-query matching.

#### `_fuzzy_match(text, pattern)`
Perform fuzzy string matching.

#### `_extract_concepts(text)`
Extract key concepts from text.

#### `_calculate_concept_similarity(concepts1, concepts2)`
Calculate similarity between concept sets.

## Usage Examples

### Basic Search
```python
from context_manager.core.memory_manager import MemoryManager

manager = MemoryManager()

# Add some memories
manager.add_memory("Machine learning is important for AI", 
                  importance=0.8, tags=["AI", "ML"])
manager.add_memory("Deep learning uses neural networks", 
                  importance=0.9, tags=["AI", "DL"])

# Search with fuzzy matching
results = manager.search_memories("machine learnin")
for memory in results:
    print(f"Found: {memory.content} (Score: {memory.importance})")
```

### Semantic Search
```python
# Find conceptually similar memories
results = manager.semantic_search("artificial intelligence")
for memory in results:
    print(f"Similar: {memory.content}")
```

### Tag-Based Search
```python
# Search by multiple tags
results = manager.search_by_tags(["AI", "learning"], match_all=False)
for memory in results:
    print(f"Tagged: {memory.content}, Tags: {memory.tags}")
```

### Time-Based Search
```python
from datetime import datetime, timedelta

# Search recent memories
start_time = datetime.now() - timedelta(days=7)
end_time = datetime.now()
results = manager.search_by_time_range(start_time, end_time)
for memory in results:
    print(f"Recent: {memory.content} ({memory.timestamp})")
```

### Memory Clustering
```python
# Get memory clusters
clusters = manager.get_memory_clusters(cluster_size=3)
for cluster_label, memories in clusters.items():
    print(f"Cluster '{cluster_label}':")
    for memory in memories:
        print(f"  - {memory.content}")
```

### Related Memories
```python
# Find related memories
target_memory = manager.short_term_memory[0]
related = manager.find_related_memories(target_memory)
for memory in related:
    print(f"Related: {memory.content}")
```

## Performance Considerations

### Search Optimization
- All search methods include early termination
- Results are pre-sorted by relevance/importance
- Configurable limits prevent memory overload
- Concept extraction is cached where possible

### Memory Management
- Automatic memory limit enforcement
- Importance-based eviction
- Efficient data structures for fast lookup
- Minimal memory overhead for search indices

### Scalability
- Linear time complexity for most operations
- Efficient set operations for concept matching
- Configurable thresholds for quality/speed tradeoffs
- Memory usage scales with content, not search capabilities

## Testing

The enhanced memory system includes comprehensive tests covering:

- Advanced search functionality
- Fuzzy matching accuracy
- Semantic search capabilities
- Tag-based search operations
- Time-based search operations
- Memory clustering
- Related memory discovery
- Error handling and edge cases
- Performance with large datasets

Run tests with:
```bash
python -m pytest tests/test_core/test_memory_manager_enhanced.py -v
```

## Integration

The enhanced memory management system is designed to integrate seamlessly with:

- **Context Management**: Provides intelligent context retrieval
- **Tool Systems**: Enables tool execution with memory awareness
- **Storage Systems**: Complements persistence with intelligent access
- **Configuration Systems**: Uses system-wide settings for thresholds

## Future Enhancements

Potential future improvements:

1. **Vector-based semantic search**: Using embeddings for better semantic understanding
2. **Machine learning relevance**: Train models for better relevance scoring
3. **Cross-memory relationships**: Discover complex relationships between memories
4. **Temporal patterns**: Identify time-based patterns in memory access
5. **Memory compression**: Intelligent compression of similar memories
6. **Distributed memory**: Support for distributed memory storage

## Configuration

Key configuration parameters:

- `fuzzy_threshold`: Default fuzzy matching threshold (0.6)
- `similarity_threshold`: Semantic search threshold (0.1)
- `cluster_size`: Default cluster size (5)
- `recency_boost_days`: Days for recency boost calculation (365)

These can be adjusted based on specific use cases and performance requirements.