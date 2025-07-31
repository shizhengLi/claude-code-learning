"""
Basic usage example for the Python Context Manager.

This example demonstrates the fundamental features of the context manager system:
- Creating and initializing the context manager
- Adding messages to the conversation
- Managing context windows
- Basic memory operations
"""

import asyncio
import tempfile
from pathlib import Path

from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig


async def basic_usage_example():
    """Demonstrate basic usage of the context manager."""
    print("=== Basic Usage Example ===")
    
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configuration
        config = ContextManagerConfig(
            max_tokens=1000,
            cache_path=temp_dir,
            short_term_memory_size=1000,
            medium_term_memory_size=2000,
            long_term_memory_size=5000,
            compression_ratio=0.8
        )
        
        # Initialize context manager
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Example 1: Basic message operations
            print("\n1. Basic Message Operations")
            print("-" * 30)
            
            # Add messages
            await context_manager.add_message("user", "Hello, I need help with Python programming.")
            await context_manager.add_message("assistant", "I'd be happy to help you with Python! What specific topic are you interested in?")
            await context_manager.add_message("user", "I'm trying to understand context managers.")
            
            # Get context summary
            summary = context_manager.get_context_summary()
            print(f"Message count: {summary['message_count']}")
            print(f"Token count: {summary['token_count']}")
            print(f"Context utilization: {summary['utilization']:.2%}")
            
            # Example 2: Context window management
            print("\n2. Context Window Management")
            print("-" * 30)
            
            # Add more messages to see context compression
            for i in range(5):
                await context_manager.add_message("user", f"Question {i+1}: How do context managers work?")
                await context_manager.add_message("assistant", f"Answer {i+1}: Context managers use __enter__ and __exit__ methods.")
            
            # Check compression history
            if context_manager.compression_history:
                print(f"Context compressed {len(context_manager.compression_history)} times")
                latest_compression = context_manager.compression_history[-1]
                print(f"Latest compression: {latest_compression['compression_ratio']:.2%} reduction")
            
            # Example 3: Memory operations
            print("\n3. Memory Operations")
            print("-" * 30)
            
            # Search memory
            memories = await context_manager.search_memory("context managers", limit=3)
            print(f"Found {len(memories)} relevant memories")
            for i, memory in enumerate(memories[:2], 1):
                print(f"  {i}. {memory.content[:100]}...")
            
            # Example 4: System status
            print("\n4. System Status")
            print("-" * 30)
            
            status = context_manager.get_system_status()
            print(f"System status: {status['status']}")
            print(f"Performance score: {status['performance'].get('health_score', 0.0):.2f}")
            
        finally:
            # Cleanup
            await context_manager.cleanup()
    
    print("\n=== Basic Usage Example Complete ===")


async def simple_context_example():
    """Simple example showing core context management."""
    print("\n=== Simple Context Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=500,
            cache_path=temp_dir,
            compression_ratio=0.7
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Simulate a conversation
            conversation = [
                ("user", "I'm learning about async programming in Python."),
                ("assistant", "Async programming allows you to write concurrent code using async/await syntax."),
                ("user", "Can you explain the difference between async and threading?"),
                ("assistant", "Async uses a single thread with an event loop, while threading uses multiple threads."),
                ("user", "What are the benefits of async programming?")
            ]
            
            # Add conversation
            for role, content in conversation:
                await context_manager.add_message(role, content)
            
            # Display current context
            print(f"Total messages: {len(context_manager.context_window.messages)}")
            print(f"Total tokens: {context_manager.context_window.current_tokens}")
            
            # Show recent messages
            print("\nRecent messages:")
            for i, msg in enumerate(context_manager.context_window.messages[-4:], 1):
                print(f"  {i}. {msg.role}: {msg.content[:60]}...")
                
        finally:
            await context_manager.cleanup()
    
    print("=== Simple Context Example Complete ===")


async def configuration_example():
    """Example showing different configuration options."""
    print("\n=== Configuration Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Different configurations for different use cases
        configs = {
            "lightweight": ContextManagerConfig(
                max_tokens=500,
                cache_path=temp_dir + "/lightweight",
                short_term_memory_size=500,
                compression_ratio=0.9
            ),
            "balanced": ContextManagerConfig(
                max_tokens=2000,
                cache_path=temp_dir + "/balanced",
                short_term_memory_size=1000,
                medium_term_memory_size=2000,
                compression_ratio=0.8
            ),
            "heavy": ContextManagerConfig(
                max_tokens=5000,
                cache_path=temp_dir + "/heavy",
                short_term_memory_size=2000,
                medium_term_memory_size=5000,
                long_term_memory_size=10000,
                compression_ratio=0.7
            )
        }
        
        for name, config in configs.items():
            print(f"\nTesting {name} configuration:")
            print(f"  Max tokens: {config.max_tokens}")
            print(f"  Compression ratio: {config.compression_ratio}")
            
            context_manager = EnhancedContextManager(config=config)
            await context_manager.initialize()
            
            # Add some test messages
            for i in range(3):
                await context_manager.add_message("user", f"Test message {i+1}")
                await context_manager.add_message("assistant", f"Test response {i+1}")
            
            summary = context_manager.get_context_summary()
            print(f"  Messages: {summary['message_count']}")
            print(f"  Utilization: {summary['utilization']:.2%}")
            
            await context_manager.cleanup()
    
    print("\n=== Configuration Example Complete ===")


async def main():
    """Run all basic usage examples."""
    print("Python Context Manager - Basic Usage Examples")
    print("=" * 50)
    
    await basic_usage_example()
    await simple_context_example()
    await configuration_example()
    
    print("\n" + "=" * 50)
    print("All basic usage examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())