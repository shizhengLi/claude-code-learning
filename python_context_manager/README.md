# Python Context Manager

A comprehensive Python implementation of Claude Code's context management and memory management system, featuring intelligent context compression, three-tier memory architecture, and extensible tool system.

## ✨ Features

### 🧠 Memory Management
- **Three-tier memory architecture**: Short-term, medium-term, and long-term memory
- **Intelligent memory search**: Content-based and tag-based search
- **Memory consolidation**: Automatic optimization of stored memories
- **Importance scoring**: Prioritization based on relevance and access patterns

### 📝 Context Management
- **Smart context compression**: Automatic context optimization when approaching token limits
- **Priority-based management**: Critical messages preserved during compression
- **Context window monitoring**: Real-time utilization tracking
- **Multi-format support**: Handles various message types and metadata

### 🛠️ Tool System
- **Extensible tool framework**: Easy creation and registration of custom tools
- **Tool chaining**: Compose multiple tools for complex workflows
- **Permission management**: Access control and rate limiting
- **Error handling**: Robust error recovery and timeout management

### 📊 Performance Monitoring
- **Real-time metrics**: CPU, memory, and operation tracking
- **Performance optimization**: Automatic system optimization
- **Health checking**: System diagnostics and recommendations
- **Benchmarking**: Built-in performance testing capabilities

### ⚡ Concurrency Support
- **Async-first design**: Built on asyncio for high performance
- **Concurrent operations**: Parallel tool execution and memory operations
- **Resource management**: Intelligent load balancing and resource allocation
- **Task scheduling**: Priority-based task execution

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/python-context-manager.git
cd python-context-manager

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
import asyncio
import tempfile
from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig

async def main():
    # Create configuration
    config = ContextManagerConfig(
        max_tokens=2000,
        compression_ratio=0.8
    )
    
    # Initialize context manager
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
            
            # Search memory
            memories = await context_manager.search_memory("hello")
            print(f"Found {len(memories)} memories")
            
        finally:
            await context_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Examples

```bash
# Run all examples
python examples/main_example.py all

# Run specific examples
python examples/main_example.py basic      # Basic usage
python examples/main_example.py advanced   # Advanced features  
python examples/main_example.py tools      # Custom tools
python examples/main_example.py performance # Performance demo

# Quick start example
python quick_start.py
```

## 📚 Documentation

- **Quick Start**: [QUICK_START.md](QUICK_START.md) - Get up and running quickly
- **User Guide**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - Comprehensive documentation
- **Examples**: [examples/README.md](examples/README.md) - Example documentation
- **Implementation Plan**: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Technical details

## 🏗️ Architecture

```
python_context_manager/
├── src/context_manager/
│   ├── core/                    # Core functionality
│   │   ├── enhanced_context_manager.py
│   │   ├── models.py
│   │   ├── config.py
│   │   ├── performance_monitor.py
│   │   ├── async_operations.py
│   │   └── health_checker.py
│   ├── memory/                  # Memory management
│   │   └── memory_manager.py
│   ├── compression/             # Context compression
│   │   └── compression_manager.py
│   ├── storage/                 # Storage system
│   │   └── hierarchical_manager.py
│   ├── tools/                   # Tool system
│   │   └── tool_manager.py
│   └── utils/                   # Utilities
│       ├── logging.py
│       ├── error_handling.py
│       └── helpers.py
├── examples/                     # Usage examples
├── tests/                       # Test suite
├── docs/                        # Documentation
└── requirements.txt
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=context_manager

# Run specific test categories
pytest tests/test_core/
pytest tests/test_integration/
pytest tests/test_performance/

# Run tests with verbose output
pytest -v
```

### Test Coverage

The project includes comprehensive test coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Benchmarking and load testing
- **End-to-End Tests**: Complete workflow testing

Current test coverage: **295 tests passing** ✅

## 🔧 Configuration

### Basic Configuration

```python
from context_manager.core.config import ContextManagerConfig

config = ContextManagerConfig(
    max_tokens=2000,                    # Maximum tokens in context
    compression_ratio=0.8,              # Target compression ratio
    cache_path="./cache",               # Cache directory
    short_term_memory_size=1000,        # Short-term memory capacity
    medium_term_memory_size=2000,       # Medium-term memory capacity
    long_term_memory_size=5000,         # Long-term memory capacity
)
```

### Environment Variables

```bash
export PYTHON_CONTEXT_MANAGER_MAX_TOKENS=2000
export PYTHON_CONTEXT_MANAGER_CACHE_PATH=/tmp/context_cache
export PYTHON_CONTEXT_MANAGER_COMPRESSION_RATIO=0.8
export PYTHON_CONTEXT_MANAGER_ENABLE_DEBUG=1
```

## 🎯 Use Cases

### 1. Chat Applications
```python
# Chat with memory and context management
await context_manager.add_message("user", user_input)
await context_manager.add_message("assistant", assistant_response)

# Search conversation history
relevant_history = await context_manager.search_memory(user_input)
```

### 2. Document Analysis
```python
# Analyze documents with custom tools
result = await context_manager.execute_tool("document_analyzer", document=text)
```

### 3. Data Processing
```python
# Process data with tool chaining
result = await context_manager.execute_tool("data_processor", data=your_data)
```

### 4. Content Generation
```python
# Generate content with context awareness
context = context_manager.get_context_summary()
generated_content = await generate_content(context)
```

## 🛠️ Custom Tools

### Creating Tools

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
```

### Using Tools

```python
# Execute tools
result = await context_manager.execute_tool(
    "text_analyzer",
    text="Hello world! This is a test."
)

if result.success:
    print(f"Analysis: {result.result}")
else:
    print(f"Error: {result.error}")
```

## 📊 Performance

### Benchmarks

- **Context Operations**: < 100ms response time
- **Memory Search**: < 50ms search time
- **Tool Execution**: Concurrent execution support
- **Memory Usage**: Efficient resource management

### Optimization Features

- **Automatic compression**: Context optimization when needed
- **Memory consolidation**: Remove redundant information
- **Performance monitoring**: Real-time metrics and alerts
- **Resource management**: Intelligent allocation and cleanup

## 🔍 Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/python-context-manager.git
cd python-context-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/ examples/
isort src/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ examples/

# Run tests
pytest
pytest --cov=context_manager
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance Monitoring

### Built-in Monitoring

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
print(f"Health score: {metrics.health_score:.2f}")
print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")

await monitor.stop_monitoring()
```

### Health Checking

```python
from context_manager.core.health_checker import HealthChecker

# Create health checker
health_checker = HealthChecker(check_interval=5.0)
await health_checker.start()

# Check system health
health_status = await health_checker.perform_health_check()
print(f"System status: {health_status.status.value}")

# Generate diagnostic report
report = await health_checker.generate_diagnostic_report()
print(f"Recommendations: {len(report.recommendations)}")
```

## 🚨 Troubleshooting

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
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or using environment variable
import os
os.environ['PYTHON_CONTEXT_MANAGER_DEBUG'] = '1'
```

## 🤝 Community

- **GitHub**: https://github.com/yourusername/python-context-manager
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Comprehensive guides and API reference

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Claude Code's architecture
- Built with modern Python async patterns
- Designed for scalability and performance

## 📞 Support

For support, please:

1. Check the [documentation](docs/)
2. Review the [examples](examples/)
3. Search existing [issues](https://github.com/yourusername/python-context-manager/issues)
4. Create a new issue with detailed information

---

**Made with ❤️ for the Python community**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-295%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-80%25-yellow.svg)](tests/)