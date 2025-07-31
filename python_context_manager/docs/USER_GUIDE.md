# Python Context Manager - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [Custom Tools](#custom-tools)
8. [Performance Optimization](#performance-optimization)
9. [Configuration](#configuration)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)

## Introduction

The Python Context Manager is a comprehensive system for managing conversation contexts, memory, and tool execution. Inspired by Claude Code's architecture, it provides:

- **Context Management**: Intelligent context window management with compression
- **Memory System**: Three-tier memory architecture (short, medium, long-term)
- **Tool System**: Extensible tool execution framework
- **Performance Monitoring**: Real-time performance metrics and optimization
- **Health Checking**: System health monitoring and diagnostics

### Key Features

- **Async-first design**: Built on asyncio for high performance
- **Modular architecture**: Clean separation of concerns
- **Extensible tools**: Easy to add custom tools
- **Smart compression**: Context optimization based on priority and relevance
- **Performance monitoring**: Built-in metrics and profiling
- **Health checks**: Automated system monitoring

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/python-context-manager.git
cd python-context-manager

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

Here's a minimal example to get you started:

```python
import asyncio
import tempfile
from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig

async def main():
    # Create configuration
    config = ContextManagerConfig(
        max_tokens=1000,
        compression_ratio=0.8
    )
    
    # Create context manager
    with tempfile.TemporaryDirectory() as temp_dir:
        config.cache_path = temp_dir
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Add messages
            await context_manager.add_message("user", "Hello, world!")
            await context_manager.add_message("assistant", "Hello! How can I help you?")
            
            # Get context summary
            summary = context_manager.get_context_summary()
            print(f"Messages: {summary['message_count']}")
            print(f"Tokens: {summary['token_count']}")
            
        finally:
            await context_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Context Management

The context manager maintains a sliding window of conversation messages with intelligent compression:

```python
# Add messages with priorities
await context_manager.add_message("user", "Important question", priority="high")
await context_manager.add_message("user", "Casual comment", priority="low")

# Context automatically compresses when token limit is reached
```

### Memory System

Three-tier memory architecture for different storage needs:

- **Short-term**: In-memory storage for immediate access
- **Medium-term**: File-based storage for recent history
- **Long-term**: Structured storage for important information

```python
# Add to memory
await context_manager.memory_manager.add_memory(
    content="Important concept",
    importance=0.8,
    tags=["important", "concept"]
)

# Search memory
memories = await context_manager.search_memory("concept", limit=5)
```

### Tool System

Extensible tool framework for executing functions:

```python
# Register a custom tool
async def my_tool(parameters):
    return {"result": "Tool executed successfully"}

context_manager.tool_manager.register_function_tool(
    "my_tool", my_tool, "Description of my tool"
)

# Execute tool
result = await context_manager.execute_tool("my_tool", param1="value1")
```

## Basic Usage

### Initialization

```python
from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig

# Create configuration
config = ContextManagerConfig(
    max_tokens=2000,           # Maximum tokens in context
    cache_path="./cache",       # Directory for cache files
    compression_ratio=0.8,     # Target compression ratio
    short_term_memory_size=1000, # Short-term memory capacity
)

# Initialize context manager
context_manager = EnhancedContextManager(config=config)
await context_manager.initialize()
```

### Message Operations

```python
# Add messages
await context_manager.add_message("user", "Hello!")
await context_manager.add_message("assistant", "Hi there!")

# Add messages with metadata
await context_manager.add_message(
    "user", 
    "Important question",
    priority="high",
    metadata={"category": "technical"}
)

# Get context summary
summary = context_manager.get_context_summary()
print(f"Messages: {summary['message_count']}")
print(f"Tokens: {summary['token_count']}")
print(f"Utilization: {summary['utilization']:.2%}")
```

### Memory Operations

```python
# Add memories with different importance levels
await context_manager.memory_manager.add_memory(
    content="Python context managers use __enter__ and __exit__",
    importance=0.9,
    tags=["python", "programming", "important"]
)

await context_manager.memory_manager.add_memory(
    content="Casual conversation about weather",
    importance=0.3,
    tags=["casual", "weather"]
)

# Search memories
results = await context_manager.search_memory("python", limit=5)
for memory in results:
    print(f"Found: {memory.content[:50]}...")

# Search by tags
tag_results = await context_manager.memory_manager.search_by_tags(["important"])
```

## Advanced Features

### Context Compression

The system automatically compresses context when approaching token limits:

```python
# Add many messages to trigger compression
for i in range(50):
    await context_manager.add_message("user", f"Long message {i}")
    await context_manager.add_message("assistant", f"Long response {i}")

# Check compression history
if context_manager.compression_history:
    latest = context_manager.compression_history[-1]
    print(f"Compressed {latest['tokens_before']} -> {latest['tokens_after']} tokens")
```

### Performance Monitoring

Built-in performance monitoring and metrics:

```python
from context_manager.core.performance_monitor import PerformanceMonitor

# Start monitoring
monitor = PerformanceMonitor(sample_interval=0.1)
await monitor.start_monitoring()

# Perform operations
await context_manager.add_message("user", "Test message")
await context_manager.search_memory("test")

# Get metrics
metrics = monitor.get_current_metrics()
print(f"Operations: {metrics.total_operations}")
print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
print(f"Health score: {metrics.health_score:.2f}")

await monitor.stop_monitoring()
```

### Concurrent Operations

High-performance concurrent operation support:

```python
from context_manager.core.async_operations import ConcurrencyController

# Create concurrency controller
controller = ConcurrencyController(max_concurrent_tasks=10)
await controller.start()

# Submit multiple tasks
task_ids = []
for i in range(5):
    task_id = await controller.submit_task(
        f"task_{i}",
        some_async_function,
        arg1, arg2,
        priority="normal"
    )
    task_ids.append(task_id)

# Wait for results
results = []
for task_id in task_ids:
    result = await controller.wait_for_task(task_id)
    results.append(result)

await controller.stop()
```

### Health Checking

Automated system health monitoring:

```python
from context_manager.core.health_checker import HealthChecker

# Create health checker
health_checker = HealthChecker(check_interval=5.0)
await health_checker.start()

# Perform health check
health_status = await health_checker.perform_health_check()
print(f"System status: {health_status.status.value}")
print(f"Health score: {health_status.health_score:.2f}")

# Generate diagnostic report
report = await health_checker.generate_diagnostic_report()
print(f"Recommendations: {len(report.recommendations)}")

await health_checker.stop()
```

## Custom Tools

### Creating Basic Tools

```python
from context_manager.core.tool_system import tool

@tool(
    name="text_analyzer",
    description="Analyze text and provide statistics",
    category=ToolCategory.ANALYSIS,
    tags=["text", "analysis"]
)
async def text_analyzer(parameters):
    """Analyze text and return statistics."""
    text = parameters.get("text", "")
    
    return {
        "word_count": len(text.split()),
        "character_count": len(text),
        "average_word_length": len(text.split()) / len(text.split()) if text.split() else 0
    }

# Register the tool
context_manager.tool_manager.register_function_tool(
    "text_analyzer", text_analyzer, "Text analysis tool"
)

# Execute the tool
result = await context_manager.execute_tool(
    "text_analyzer",
    text="Hello world! This is a test."
)
```

### Advanced Tool Features

```python
@tool(
    name="data_processor",
    description="Process and transform data",
    category=ToolCategory.DATA_PROCESSING,
    timeout=30.0,  # Tool timeout
    permission=ToolPermission.RESTRICTED,  # Access control
    rate_limit=10,  # Rate limiting
    rate_window=60.0
)
async def data_processor(parameters):
    """Process data with error handling and validation."""
    try:
        data = parameters.get("data", {})
        
        # Validate input
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        # Process data
        processed = {k.upper(): v for k, v in data.items()}
        
        return {
            "success": True,
            "processed_data": processed,
            "original_size": len(data),
            "processed_size": len(processed)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Tool Chaining

```python
# Execute multiple tools in sequence
async def analyze_and_summarize(text):
    """Chain multiple tools together."""
    
    # Step 1: Analyze text
    analysis_result = await context_manager.execute_tool(
        "text_analyzer",
        text=text
    )
    
    if not analysis_result.success:
        return {"error": "Analysis failed"}
    
    # Step 2: Create summary
    summary_result = await context_manager.execute_tool(
        "text_summarizer",
        text=text,
        max_length=100
    )
    
    return {
        "analysis": analysis_result.result,
        "summary": summary_result.result if summary_result.success else None
    }
```

## Performance Optimization

### Configuration Optimization

Different configurations for different use cases:

```python
# Lightweight configuration for simple applications
lightweight_config = ContextManagerConfig(
    max_tokens=500,
    compression_ratio=0.9,
    short_term_memory_size=500
)

# Balanced configuration for general use
balanced_config = ContextManagerConfig(
    max_tokens=2000,
    compression_ratio=0.8,
    short_term_memory_size=1000,
    medium_term_memory_size=2000
)

# Heavy configuration for complex applications
heavy_config = ContextManagerConfig(
    max_tokens=5000,
    compression_ratio=0.6,
    short_term_memory_size=2000,
    medium_term_memory_size=5000,
    long_term_memory_size=10000
)
```

### Memory Management

```python
# Optimize memory usage
await context_manager.optimize_system()

# Clear old memories
await context_manager.memory_manager.cleanup_old_memories(max_age_days=30)

# Consolidate similar memories
consolidation_result = await context_manager.memory_manager.consolidate_memories()
print(f"Consolidated {consolidation_result['consolidated_count']} memories")
```

### Performance Monitoring

```python
# Enable detailed monitoring
monitor = PerformanceMonitor(
    sample_interval=0.05,  # Frequent sampling
    enable_profiling=True  # Enable profiling
)

await monitor.start_monitoring()

# Monitor specific operations
with monitor.operation_timer("custom_operation"):
    await perform_custom_operation()

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Health score: {summary['health_score']:.2f}")
print(f"Recommendations: {summary['recommendations']}")
```

## Configuration

### Configuration Options

```python
config = ContextManagerConfig(
    # Context settings
    max_tokens=2000,                    # Maximum tokens in context
    compression_ratio=0.8,              # Target compression ratio
    
    # Memory settings
    short_term_memory_size=1000,        # Short-term memory capacity
    medium_term_memory_size=2000,       # Medium-term memory capacity
    long_term_memory_size=5000,         # Long-term memory capacity
    
    # Storage settings
    cache_path="./cache",               # Cache directory
    storage_path="./storage",           # Storage directory
    
    # Performance settings
    enable_compression=True,            # Enable context compression
    enable_memory_index=True,           # Enable memory indexing
    enable_performance_monitor=True,    # Enable performance monitoring
    
    # Tool settings
    max_tool_timeout=30.0,             # Maximum tool execution time
    enable_tool_caching=True,           # Enable tool result caching
    
    # Health check settings
    health_check_interval=60.0,         # Health check interval
    enable_auto_recovery=True           # Enable automatic recovery
)
```

### Environment Variables

Configuration can be overridden with environment variables:

```bash
export PYTHON_CONTEXT_MANAGER_MAX_TOKENS=2000
export PYTHON_CONTEXT_MANAGER_CACHE_PATH=/tmp/context_cache
export PYTHON_CONTEXT_MANAGER_COMPRESSION_RATIO=0.8
export PYTHON_CONTEXT_MANAGER_ENABLE_DEBUG=1
```

## Examples

The project includes comprehensive examples in the `examples/` directory:

### Running Examples

```bash
# Run all examples
python examples/main_example.py all

# Run specific examples
python examples/main_example.py basic
python examples/main_example.py advanced
python examples/main_example.py tools
python examples/main_example.py performance

# Run with verbose output
python examples/main_example.py basic --verbose
```

### Example Categories

1. **Basic Usage** (`basic_usage.py`): Core functionality and basic operations
2. **Advanced Features** (`advanced_features.py`): Advanced features and optimization
3. **Custom Tools** (`custom_tools.py`): Tool creation and management
4. **Performance Demo** (`performance_demo.py`): Performance testing and benchmarking

## Troubleshooting

### Common Issues

#### Import Errors

```python
# Ensure the src directory is in your path
import sys
sys.path.insert(0, "./src")

from context_manager.core.enhanced_context_manager import EnhancedContextManager
```

#### Memory Issues

```python
# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f}MB")

# Optimize memory
await context_manager.optimize_system()
```

#### Performance Issues

```python
# Enable performance monitoring
monitor = PerformanceMonitor()
await monitor.start_monitoring()

# Check performance metrics
metrics = monitor.get_current_metrics()
print(f"Health score: {metrics.health_score:.2f}")

# Get optimization recommendations
summary = monitor.get_performance_summary()
for recommendation in summary['recommendations']:
    print(f"Recommendation: {recommendation}")
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or using environment variable
import os
os.environ['PYTHON_CONTEXT_MANAGER_DEBUG'] = '1'
```

### Error Handling

```python
try:
    await context_manager.add_message("user", "Hello")
except ContextManagerError as e:
    print(f"Context manager error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## API Reference

### Core Classes

#### EnhancedContextManager

Main class for managing context and memory operations.

```python
class EnhancedContextManager:
    async def initialize(self) -> None
    async def cleanup(self) -> None
    async def add_message(self, role: str, content: str, **kwargs) -> bool
    async def search_memory(self, query: str, limit: int = 10) -> List[Memory]
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult
    def get_context_summary(self) -> Dict[str, Any]
    def get_system_status(self) -> Dict[str, Any]
```

#### ContextManagerConfig

Configuration class for context manager settings.

```python
class ContextManagerConfig:
    def __init__(
        self,
        max_tokens: int = 2000,
        compression_ratio: float = 0.8,
        cache_path: str = "./cache",
        # ... other parameters
    )
```

### Tool System

#### Tool Registration

```python
@tool(name="my_tool", description="Tool description")
async def my_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
    # Tool implementation
    pass

# Register tool
context_manager.tool_manager.register_function_tool(
    "my_tool", my_tool, "Tool description"
)
```

#### Tool Execution

```python
result = await context_manager.execute_tool(
    "tool_name",
    param1="value1",
    param2="value2"
)

if result.success:
    print(f"Tool result: {result.result}")
else:
    print(f"Tool error: {result.error}")
```

### Performance Monitoring

#### PerformanceMonitor

```python
monitor = PerformanceMonitor(sample_interval=0.1)
await monitor.start_monitoring()

# Record operations
monitor.record_operation("operation_name", 0.1, True)

# Get metrics
metrics = monitor.get_current_metrics()
summary = monitor.get_performance_summary()

await monitor.stop_monitoring()
```

### Memory Management

#### Memory Operations

```python
# Add memory
await context_manager.memory_manager.add_memory(
    content="Memory content",
    importance=0.5,
    tags=["tag1", "tag2"]
)

# Search memory
memories = await context_manager.search_memory("query", limit=10)

# Search by tags
memories = await context_manager.memory_manager.search_by_tags(["tag1"])

# Get memory statistics
stats = await context_manager.memory_manager.get_memory_stats()
```

## Best Practices

### Performance

1. **Use appropriate configuration**: Choose configuration based on your use case
2. **Monitor performance**: Use built-in monitoring to track system health
3. **Optimize regularly**: Run optimization tasks periodically
4. **Handle errors gracefully**: Implement proper error handling

### Memory Management

1. **Set appropriate importance levels**: Use high importance for critical information
2. **Use tags effectively**: Organize memories with descriptive tags
3. **Clean up regularly**: Remove old or irrelevant memories
4. **Monitor memory usage**: Keep track of memory consumption

### Tool Development

1. **Validate inputs**: Always validate tool parameters
2. **Handle errors**: Provide meaningful error messages
3. **Use timeouts**: Set appropriate timeouts for long-running operations
4. **Document tools**: Provide clear descriptions and examples

### Concurrency

1. **Use appropriate concurrency levels**: Don't overload the system
2. **Monitor resource usage**: Track CPU and memory usage
3. **Handle timeouts**: Implement proper timeout handling
4. **Use rate limiting**: Prevent abuse of resources

## Contributing

We welcome contributions to the Python Context Manager project! Please see the main repository for contribution guidelines.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For support, please:

1. Check the documentation
2. Review the examples
3. Search existing issues
4. Create a new issue with detailed information

---

*Last updated: 2025-07-31*