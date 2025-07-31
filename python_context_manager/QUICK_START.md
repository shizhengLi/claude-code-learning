# Quick Start Guide

This guide will help you get up and running with the Python Context Manager quickly.

## Prerequisites

- Python 3.11+
- pip package manager

## Installation

### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/python-context-manager.git
cd python-context-manager

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Dependencies Only

```bash
# Just install the required dependencies
pip install -r requirements.txt
```

## Your First Context Manager

Create a file `quick_start.py`:

```python
import asyncio
import tempfile
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig

async def main():
    """Quick start example."""
    print("🚀 Python Context Manager - Quick Start")
    print("=" * 50)
    
    # Create configuration
    config = ContextManagerConfig(
        max_tokens=1000,
        compression_ratio=0.8
    )
    
    # Use temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        config.cache_path = temp_dir
        
        # Initialize context manager
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Step 1: Add some messages
            print("\n1. Adding messages to conversation...")
            await context_manager.add_message("user", "Hello, I'm learning about context managers!")
            await context_manager.add_message("assistant", "That's great! Context managers are powerful tools in Python.")
            await context_manager.add_message("user", "Can you explain how they work?")
            await context_manager.add_message("assistant", "Context managers use the `with` statement and implement `__enter__` and `__exit__` methods.")
            
            # Step 2: Check context status
            print("\n2. Checking context status...")
            summary = context_manager.get_context_summary()
            print(f"   Messages: {summary['message_count']}")
            print(f"   Tokens: {summary['token_count']}")
            print(f"   Context utilization: {summary['utilization']:.1%}")
            
            # Step 3: Add some important information to memory
            print("\n3. Storing important information in memory...")
            await context_manager.memory_manager.add_memory(
                content="Context managers automatically handle resource cleanup",
                importance=0.9,
                tags=["python", "important", "concept"]
            )
            
            await context_manager.memory_manager.add_memory(
                content="The `with` statement ensures proper setup and teardown",
                importance=0.8,
                tags=["python", "syntax", "important"]
            )
            
            # Step 4: Search memory
            print("\n4. Searching memory for information...")
            memories = await context_manager.search_memory("context manager", limit=3)
            print(f"   Found {len(memories)} relevant memories")
            for i, memory in enumerate(memories, 1):
                print(f"   {i}. {memory.content[:60]}...")
            
            # Step 5: Create and use a custom tool
            print("\n5. Creating and using a custom tool...")
            
            async def text_analyzer(parameters):
                """Simple text analysis tool."""
                text = parameters.get("text", "")
                words = text.split()
                return {
                    "word_count": len(words),
                    "character_count": len(text),
                    "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
                }
            
            # Register the tool
            context_manager.tool_manager.register_function_tool(
                "text_analyzer", text_analyzer, "Analyze text statistics"
            )
            
            # Use the tool
            result = await context_manager.execute_tool(
                "text_analyzer",
                text="Context managers are very useful for resource management in Python!"
            )
            
            if result.success:
                analysis = result.result
                print(f"   Word count: {analysis['word_count']}")
                print(f"   Character count: {analysis['character_count']}")
                print(f"   Average word length: {analysis['average_word_length']:.1f}")
            
            # Step 6: Check system health
            print("\n6. Checking system health...")
            status = context_manager.get_system_status()
            print(f"   System status: {status['status']}")
            print(f"   Performance health: {status['performance']['health_score']:.2f}")
            
            # Step 7: Show system optimization
            print("\n7. Optimizing system...")
            optimization = await context_manager.optimize_system()
            print(f"   Actions taken: {len(optimization.get('actions_taken', []))}")
            
            print("\n✅ Quick start completed successfully!")
            print("\nNext steps:")
            print("   - Run the examples: python examples/main_example.py all")
            print("   - Read the user guide: docs/USER_GUIDE.md")
            print("   - Explore the API documentation")
            
        finally:
            # Clean up
            await context_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Run the Quick Start

```bash
python quick_start.py
```

## Expected Output

```
🚀 Python Context Manager - Quick Start
==================================================

1. Adding messages to conversation...

2. Checking context status...
   Messages: 4
   Tokens: 87
   Context utilization: 8.7%

3. Storing important information in memory...

4. Searching memory for information...
   Found 2 relevant memories
   1. Context managers automatically handle resource cleanup...
   2. The `with` statement ensures proper setup and teardown...

5. Creating and using a custom tool...
   Word count: 11
   Character count: 68
   Average word length: 6.2

6. Checking system health...
   System status: ready
   Performance health: 1.00

7. Optimizing system...
   Actions taken: 2

✅ Quick start completed successfully!

Next steps:
   - Run the examples: python examples/main_example.py all
   - Read the user guide: docs/USER_GUIDE.md
   - Explore the API documentation
```

## Next Steps

### 1. Run the Examples

The project includes comprehensive examples:

```bash
# Run all examples
python examples/main_example.py all

# Run specific examples
python examples/main_example.py basic      # Basic usage
python examples/main_example.py advanced   # Advanced features
python examples/main_example.py tools      # Custom tools
python examples/main_example.py performance # Performance demo
```

### 2. Read the Documentation

- **User Guide**: `docs/USER_GUIDE.md` - Comprehensive guide
- **Examples**: `examples/README.md` - Example documentation
- **Implementation Plan**: `IMPLEMENTATION_PLAN.md` - Technical details

### 3. Try Your Own Use Case

```python
import asyncio
import tempfile
from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig

async def my_use_case():
    config = ContextManagerConfig(max_tokens=2000)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config.cache_path = temp_dir
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Your custom code here
            await context_manager.add_message("user", "Your message here")
            
            # Search memory
            memories = await context_manager.search_memory("your query")
            
            # Execute tools
            result = await context_manager.execute_tool("your_tool")
            
        finally:
            await context_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(my_use_case())
```

### 4. Explore the API

Key classes and methods:

- `EnhancedContextManager`: Main context management class
- `ContextManagerConfig`: Configuration settings
- `PerformanceMonitor`: Performance monitoring
- `HealthChecker`: System health monitoring
- `ConcurrencyController`: Concurrent operations

### 5. Common Use Cases

#### Chat Application
```python
# Simple chat with memory
await context_manager.add_message("user", user_input)
await context_manager.add_message("assistant", assistant_response)

# Search conversation history
relevant_history = await context_manager.search_memory(user_input)
```

#### Document Analysis
```python
# Analyze documents with tools
result = await context_manager.execute_tool("document_analyzer", document=text)
```

#### Data Processing
```python
# Process data with custom tools
result = await context_manager.execute_tool("data_processor", data=your_data)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to add the `src` directory to your Python path
2. **Permission Error**: Ensure the cache directory is writable
3. **Async Error**: All operations are async - use `await`

### Get Help

- Check the user guide: `docs/USER_GUIDE.md`
- Run examples: `python examples/main_example.py all`
- Enable debug mode: Set environment variable `PYTHON_CONTEXT_MANAGER_DEBUG=1`

## Resources

- **GitHub Repository**: https://github.com/yourusername/python-context-manager
- **Documentation**: docs/ directory
- **Examples**: examples/ directory
- **Issues**: Report bugs and request features

Happy coding with the Python Context Manager! 🎉