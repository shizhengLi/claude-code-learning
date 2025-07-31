# Python Context Manager Examples

This directory contains comprehensive examples demonstrating the capabilities of the Python Context Manager system. Each example focuses on different aspects of the system and can be run independently.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

**Focus**: Core functionality and basic operations

**Topics covered**:
- Context manager initialization and configuration
- Basic message operations
- Context window management
- Memory operations
- System status monitoring
- Different configuration options

**Run time**: 5-10 minutes

```bash
python basic_usage.py
# or
python main_example.py basic
```

### 2. Advanced Features (`advanced_features.py`)

**Focus**: Advanced features and optimization techniques

**Topics covered**:
- Advanced memory management and search
- Context compression and optimization
- Performance monitoring and metrics
- Concurrent operations
- Health checking and diagnostics
- Error handling and recovery

**Run time**: 15-20 minutes

```bash
python advanced_features.py
# or
python main_example.py advanced
```

### 3. Custom Tools (`custom_tools.py`)

**Focus**: Tool creation, registration, and management

**Topics covered**:
- Creating custom tool functions
- Tool registration and discovery
- Tool execution with parameters
- Tool chaining and composition
- Error handling in tools
- Tool permissions and security
- Tool statistics and monitoring

**Run time**: 10-15 minutes

```bash
python custom_tools.py
# or
python main_example.py tools
```

### 4. Performance Demo (`performance_demo.py`)

**Focus**: Performance testing, benchmarking, and optimization

**Topics covered**:
- Performance benchmarking
- Load testing and stress testing
- Memory usage analysis
- Optimization comparison
- Performance monitoring
- Real-time metrics collection

**Run time**: 10-15 minutes

```bash
python performance_demo.py
# or
python main_example.py performance
```

## Quick Start

### Running Individual Examples

Each example can be run independently:

```bash
# Run basic usage examples
python examples/basic_usage.py

# Run advanced features
python examples/advanced_features.py

# Run custom tools examples
python examples/custom_tools.py

# Run performance demos
python examples/performance_demo.py
```

### Using the Main Example Runner

The main example runner provides a convenient interface:

```bash
# Run all examples
python examples/main_example.py all

# Run specific examples
python examples/main_example.py basic
python examples/main_example.py advanced
python examples/main_example.py tools
python examples/main_example.py performance

# Show example information
python examples/main_example.py info

# Enable verbose output
python examples/main_example.py basic --verbose
```

### Running from Python

You can also import and run examples programmatically:

```python
from examples import run_basic_examples, run_advanced_examples

import asyncio

# Run basic examples
asyncio.run(run_basic_examples())

# Run advanced examples
asyncio.run(run_advanced_examples())
```

## Prerequisites

All examples require:

- Python 3.11+
- The Python Context Manager package installed
- Dependencies listed in `requirements.txt`

To install dependencies:

```bash
pip install -r requirements.txt
```

## Example Dependencies

Some examples have additional requirements:

### Performance Demo
- `psutil` for system monitoring
- `tracemalloc` for memory tracking (built-in)

### Custom Tools
- All tools use the built-in context manager tool system

## Configuration

Examples use temporary directories for data storage to avoid cluttering your system. You can modify the configuration in each example to use persistent storage if needed.

```python
# Example configuration modification
config = ContextManagerConfig(
    max_tokens=2000,
    cache_path="/path/to/your/cache",  # Change this
    compression_ratio=0.8
)
```

## Output and Logs

Examples produce detailed output showing:

- System initialization and configuration
- Operation results and performance metrics
- Error messages and recovery information
- Performance summaries and recommendations

### Performance Metrics

Performance examples include:

- Execution times
- Memory usage statistics
- Cache hit/miss ratios
- System health scores
- Throughput measurements

## Common Use Cases

### Learning the System

Start with `basic_usage.py` to understand core concepts, then progress to `advanced_features.py` for deeper functionality.

### Testing Custom Tools

Use `custom_tools.py` as a template for creating your own tools and understanding the tool system.

### Performance Optimization

Run `performance_demo.py` to understand performance characteristics and optimization opportunities.

### Integration Testing

Examples can be modified to test specific integration scenarios with your applications.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in your Python path
2. **Permission Errors**: Examples use temporary directories, but ensure write permissions
3. **Memory Issues**: Some performance examples may use significant memory
4. **Async Errors**: All examples use async/await patterns

### Debug Mode

Add verbose output to get more detailed information:

```bash
python main_example.py basic --verbose
```

### Environment Variables

Some behavior can be controlled with environment variables:

```bash
# Enable debug logging
export PYTHON_CONTEXT_MANAGER_DEBUG=1

# Set custom cache directory
export PYTHON_CONTEXT_MANAGER_CACHE=/tmp/my_cache

# Run examples
python examples/basic_usage.py
```

## Extending Examples

### Creating Custom Examples

Use existing examples as templates:

```python
import asyncio
import tempfile
from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig

async def my_custom_example():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(cache_path=temp_dir)
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Your custom code here
            pass
        finally:
            await context_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(my_custom_example())
```

### Adding New Tools

See `custom_tools.py` for examples of creating and registering tools.

### Performance Testing

Use `performance_demo.py` as a starting point for custom performance tests.

## Contributing

When contributing new examples:

1. Follow the existing code style and structure
2. Include comprehensive comments and documentation
3. Handle errors gracefully
4. Use temporary directories for file operations
5. Include performance metrics where relevant
6. Test your examples thoroughly

## Support

For issues or questions about the examples:

1. Check the main project documentation
2. Review existing examples for similar patterns
3. Ensure all dependencies are installed
4. Run with verbose output for debugging

## License

These examples are part of the Python Context Manager project and are subject to the same license terms.