# Python Context Manager

A Python implementation of Claude Code's context management and memory management system.

## Features

- **Three-tier memory architecture**: Short-term, medium-term, and long-term memory
- **Context management**: Intelligent context window management with compression
- **Tool integration**: Extensible tool system for external operations
- **Configuration management**: Flexible configuration system with environment variable support
- **Storage system**: Multiple storage backends (Redis, file system, database)
- **Caching**: Intelligent caching system for performance optimization
- **Logging and monitoring**: Comprehensive logging and error handling

## Installation

```bash
pip install python-context-manager
```

## Quick Start

```python
from context_manager import ContextManager

# Initialize context manager
context_manager = ContextManager()

# Add messages to context
context_manager.add_message("user", "Hello, world!")
context_manager.add_message("assistant", "Hello! How can I help you?")

# Get context summary
summary = context_manager.get_context_summary()
print(summary)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=context_manager

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## License

MIT License