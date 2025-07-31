"""
Python Context Manager Examples

This package contains comprehensive examples demonstrating the capabilities of the Python Context Manager system.

Examples included:
- basic_usage.py: Basic functionality and core features
- advanced_features.py: Advanced features and optimization
- custom_tools.py: Tool creation and management
- performance_demo.py: Performance testing and benchmarking

Each example can be run independently or imported for learning purposes.
"""

__version__ = "0.1.0"
__author__ = "Python Context Manager Team"
__email__ = "team@python-context-manager.dev"

# Example runners
from .basic_usage import main as run_basic_examples
from .advanced_features import main as run_advanced_examples
from .custom_tools import main as run_custom_tools_examples
from .performance_demo import main as run_performance_examples

def run_all_examples():
    """Run all examples in sequence."""
    import asyncio
    
    print("Python Context Manager - Complete Example Suite")
    print("=" * 60)
    
    async def _run_all():
        try:
            await run_basic_examples()
            await run_advanced_examples()
            await run_custom_tools_examples()
            await run_performance_examples()
        except Exception as e:
            print(f"Error running examples: {e}")
            raise
    
    asyncio.run(_run_all())

def get_example_info():
    """Get information about available examples."""
    return {
        "basic_usage": {
            "description": "Basic functionality and core features",
            "topics": ["context management", "message operations", "configuration", "system status"],
            "estimated_time": "5-10 minutes"
        },
        "advanced_features": {
            "description": "Advanced features and optimization techniques",
            "topics": ["memory management", "context compression", "performance monitoring", "concurrent operations", "health checking"],
            "estimated_time": "15-20 minutes"
        },
        "custom_tools": {
            "description": "Tool creation and management",
            "topics": ["custom tools", "tool registration", "tool chaining", "error handling", "tool permissions"],
            "estimated_time": "10-15 minutes"
        },
        "performance_demo": {
            "description": "Performance testing and benchmarking",
            "topics": ["performance benchmarking", "load testing", "memory analysis", "optimization comparison"],
            "estimated_time": "10-15 minutes"
        }
    }

__all__ = [
    "run_basic_examples",
    "run_advanced_examples", 
    "run_custom_tools_examples",
    "run_performance_examples",
    "run_all_examples",
    "get_example_info"
]