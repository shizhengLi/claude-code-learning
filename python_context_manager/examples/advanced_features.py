"""
Advanced features example for the Python Context Manager.

This example demonstrates advanced features including:
- Memory management and search
- Context compression and optimization
- Performance monitoring
- Health checking
- Error handling and recovery
- Concurrent operations
"""

import asyncio
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig
from context_manager.core.performance_monitor import PerformanceMonitor
from context_manager.core.async_operations import ConcurrencyController, AsyncOperationManager
from context_manager.core.health_checker import HealthChecker


async def memory_management_example():
    """Demonstrate advanced memory management features."""
    print("=== Advanced Memory Management Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir,
            short_term_memory_size=1000,
            medium_term_memory_size=3000,
            long_term_memory_size=5000,
            compression_ratio=0.8
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Add diverse content to memory
            topics = [
                "Python programming best practices",
                "Machine learning algorithms",
                "Data structures and algorithms",
                "Web development with Django",
                "Database design patterns",
                "API design principles",
                "Cloud computing basics",
                "DevOps practices",
                "Container technology",
                "Microservices architecture"
            ]
            
            print("\n1. Adding diverse content to memory")
            print("-" * 40)
            
            for i, topic in enumerate(topics):
                await context_manager.add_message("user", f"Tell me about {topic}")
                await context_manager.add_message("assistant", f"Here's what I know about {topic}: This is a comprehensive overview of the topic with key concepts and best practices.")
                
                # Add some metadata to make memories more interesting
                if i % 2 == 0:
                    context_manager.memory_manager.add_memory(
                        content=f"Important concept: {topic}",
                        importance=0.8,
                        tags=["important", "concept", topic.split()[0].lower()]
                    )
            
            # Advanced memory search
            print("\n2. Advanced memory search")
            print("-" * 40)
            
            # Search by content
            search_results = await context_manager.search_memory("Python", limit=5)
            print(f"Found {len(search_results)} memories related to 'Python'")
            
            # Search by tags
            tag_results = context_manager.memory_manager.search_by_tags(["important"], limit=3)
            print(f"Found {len(tag_results)} important memories")
            
            # Semantic search (if available)
            try:
                semantic_results = context_manager.memory_manager.semantic_search("programming concepts", limit=3)
                print(f"Found {len(semantic_results)} semantically similar memories")
            except AttributeError:
                print("Semantic search not available in this version")
            
            # Memory statistics
            print("\n3. Memory statistics")
            print("-" * 40)
            
            stats = context_manager.memory_manager.get_memory_stats()
            total_memories = stats['short_term']['count'] + stats['medium_term']['count'] + stats['long_term']['count']
            print(f"Total memories: {total_memories}")
            print(f"Average importance: {stats.get('average_importance', 0.0):.2f}")
            print(f"Most accessed: {stats.get('most_accessed', [])[:3] if stats.get('most_accessed') else 'N/A'}")
            
            # Memory consolidation
            print("\n4. Memory consolidation")
            print("-" * 40)
            
            print("Memory consolidation not available in this version")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Advanced Memory Management Example Complete ===")


async def context_compression_example():
    """Demonstrate advanced context compression features."""
    print("\n=== Advanced Context Compression Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=1000,  # Small limit to trigger compression
            cache_path=temp_dir,
            compression_ratio=0.6  # Aggressive compression
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Add many messages to trigger compression
            print("\n1. Adding messages to trigger compression")
            print("-" * 50)
            
            long_conversation = []
            for i in range(20):
                long_conversation.append(("user", f"This is a very long user message number {i+1} that contains a lot of detailed information about various topics and should trigger the context compression mechanism when the token limit is exceeded."))
                long_conversation.append(("assistant", f"This is an equally long assistant response number {i+1} that provides comprehensive information and detailed explanations to help the user understand the concepts being discussed."))
            
            # Add messages and monitor compression
            for role, content in long_conversation:
                await context_manager.add_message(role, content)
                
                # Check if compression occurred
                if len(context_manager.compression_history) > 0:
                    latest = context_manager.compression_history[-1]
                    print(f"Compression triggered! Ratio: {latest['compression_ratio']:.2%}")
                    break
            
            # Analyze compression history
            print("\n2. Compression analysis")
            print("-" * 50)
            
            if context_manager.compression_history:
                total_compressions = len(context_manager.compression_history)
                avg_ratio = sum(c['compression_ratio'] for c in context_manager.compression_history) / total_compressions
                print(f"Total compressions: {total_compressions}")
                print(f"Average compression ratio: {avg_ratio:.2%}")
                
                # Show most recent compression details
                latest = context_manager.compression_history[-1]
                print(f"Latest compression:")
                print(f"  Tokens before: {latest['original_tokens']}")
                print(f"  Tokens after: {latest['compressed_tokens']}")
                print(f"  Messages removed: {latest['original_count'] - latest['compressed_count']}")
            
            # Priority-based compression
            print("\n3. Priority-based compression")
            print("-" * 50)
            
            # Add high-priority messages
            await context_manager.add_message("system", "CRITICAL: System maintenance scheduled", priority="critical")
            await context_manager.add_message("user", "This is an important question about the system", priority="high")
            await context_manager.add_message("user", "This is just a casual conversation", priority="low")
            
            # Check which messages survive compression
            current_messages = context_manager.context_window.messages
            critical_count = sum(1 for msg in current_messages if msg.priority == "critical")
            high_count = sum(1 for msg in current_messages if msg.priority == "high")
            low_count = sum(1 for msg in current_messages if msg.priority == "low")
            
            print(f"Critical messages preserved: {critical_count}")
            print(f"High priority messages preserved: {high_count}")
            print(f"Low priority messages preserved: {low_count}")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Advanced Context Compression Example Complete ===")


async def performance_monitoring_example():
    """Demonstrate performance monitoring features."""
    print("\n=== Performance Monitoring Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        performance_monitor = PerformanceMonitor(sample_interval=0.5)
        
        await context_manager.initialize()
        await performance_monitor.start_monitoring()
        
        try:
            # Perform various operations while monitoring
            print("\n1. Performing operations under monitoring")
            print("-" * 50)
            
            # Record some operations
            start_time = time.time()
            await context_manager.add_message("user", "Hello!")
            performance_monitor.record_operation("message_add", time.time() - start_time, True)
            
            # Memory search operation
            start_time = time.time()
            await context_manager.search_memory("test")
            performance_monitor.record_operation("memory_search", time.time() - start_time, True)
            
            # Cache operations
            performance_monitor.record_cache_operation(hit=True)
            performance_monitor.record_cache_operation(hit=False)
            performance_monitor.record_cache_operation(hit=True)
            
            # Simulate some errors (reduced to avoid alerts)
            performance_monitor.record_operation("error_test", 0.1, False)
            performance_monitor.record_operation("success_test", 0.1, True)
            performance_monitor.record_operation("success_test", 0.1, True)
            
            # Wait for monitoring to process
            await asyncio.sleep(0.5)
            
            # Get performance metrics
            print("\n2. Performance metrics")
            print("-" * 50)
            
            metrics = performance_monitor.get_current_metrics()
            print(f"Message operations: {metrics.message_operations}")
            print(f"Memory operations: {metrics.memory_operations}")
            print(f"Cache hits: {metrics.cache_hits}")
            print(f"Cache misses: {metrics.cache_misses}")
            print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
            print(f"Error count: {metrics.error_count}")
            print(f"Health score: {performance_monitor._calculate_health_score(metrics):.2f}")
            
            # Performance summary
            print("\n3. Performance summary")
            print("-" * 50)
            
            summary = performance_monitor.get_performance_summary()
            print(f"Overall health: {summary['health_score']:.2f}")
            print(f"Recommendations: {len(summary['recommendations'])}")
            print(f"Active alerts: {summary['active_alerts']}")
            
            if summary['recommendations']:
                print("Recommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
            
            # Performance trends
            print("\n4. Performance trends")
            print("-" * 50)
            
            print("Performance trends analysis not available in this version")
            
        finally:
            await performance_monitor.stop_monitoring()
            await context_manager.cleanup()
    
    print("=== Performance Monitoring Example Complete ===")


async def concurrent_operations_example():
    """Demonstrate concurrent operations with the context manager."""
    print("\n=== Concurrent Operations Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        concurrency_controller = ConcurrencyController(max_concurrent_tasks=5)
        
        await context_manager.initialize()
        await concurrency_controller.start()
        
        try:
            # Define concurrent tasks
            async def user_session(user_id: int, num_messages: int):
                """Simulate a user session."""
                session_results = []
                
                for i in range(num_messages):
                    # Add message
                    await context_manager.add_message("user", f"User {user_id} message {i}")
                    session_results.append(f"User {user_id} added message {i}")
                    
                    # Search memory
                    memories = await context_manager.search_memory(f"User {user_id}")
                    session_results.append(f"User {user_id} found {len(memories)} memories")
                    
                    # Small delay
                    await asyncio.sleep(0.01)
                
                return session_results
            
            # Submit multiple concurrent sessions
            print("\n1. Starting concurrent user sessions")
            print("-" * 50)
            
            task_ids = []
            for user_id in range(3):
                task_id = await concurrency_controller.submit_task(
                    f"user_session_{user_id}",
                    user_session,
                    user_id,
                    5,  # 5 messages per session
                    priority="normal"
                )
                task_ids.append(task_id)
            
            # Monitor progress
            print("Monitoring concurrent operations...")
            from context_manager.core.async_operations import TaskStatus
            while not all(concurrency_controller.get_task_status(tid) in [TaskStatus.COMPLETED, TaskStatus.FAILED] for tid in task_ids):
                resource_usage = concurrency_controller.get_resource_usage()
                print(f"  Active tasks: {resource_usage.active_tasks}")
                print(f"  Queued tasks: {resource_usage.queued_tasks}")
                await asyncio.sleep(0.5)
            
            # Collect results
            print("\n2. Collecting results")
            print("-" * 50)
            
            all_results = []
            for task_id in task_ids:
                result = await concurrency_controller.wait_for_task(task_id)
                all_results.extend(result)
            
            print(f"Total operations completed: {len(all_results)}")
            
            # Check system status after concurrent operations
            print("\n3. System status after concurrent operations")
            print("-" * 50)
            
            summary = context_manager.get_context_summary()
            print(f"Total messages: {summary['message_count']}")
            print(f"Token usage: {summary['token_count']}")
            print(f"Context utilization: {summary['utilization']:.2%}")
            
        finally:
            await concurrency_controller.stop()
            await context_manager.cleanup()
    
    print("=== Concurrent Operations Example Complete ===")


async def health_checking_example():
    """Demonstrate health checking and diagnostic features."""
    print("\n=== Health Checking Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=1000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        health_checker = HealthChecker(check_interval=0.1)
        
        await context_manager.initialize()
        await health_checker.start()
        
        try:
            # Perform initial health check
            print("\n1. Initial health check")
            print("-" * 40)
            
            health_status = await health_checker.perform_health_check()
            print(f"System status: {health_status.status.value}")
            print(f"Health score: {health_status.health_score:.2f}")
            print(f"Number of checks: {len(health_status.checks)}")
            
            # Show individual checks
            for check in health_status.checks:
                print(f"  {check.name}: {check.status} ({check.message})")
            
            # Generate diagnostic report
            print("\n2. Diagnostic report")
            print("-" * 40)
            
            report = await health_checker.generate_diagnostic_report()
            report_dict = report.to_dict()
            
            print(f"Report generated: {report_dict['timestamp']}")
            print(f"System info: {len(report_dict['system_info'])} items")
            print(f"Performance metrics: {len(report_dict['performance_metrics'])} items")
            
            if report_dict['recommendations']:
                print("Recommendations:")
                for rec in report_dict['recommendations']:
                    print(f"  - {rec}")
            
            # Register custom health check
            print("\n3. Custom health check")
            print("-" * 40)
            
            async def custom_memory_check():
                """Custom health check for memory system."""
                memory_stats = context_manager.memory_manager.get_memory_stats()
                total_memories = memory_stats.get('total_memories', 0)
                
                if total_memories > 1000:
                    return {
                        'status': 'warning',
                        'message': f'High memory usage: {total_memories} memories',
                        'details': {'total_memories': total_memories}
                    }
                else:
                    return {
                        'status': 'healthy',
                        'message': f'Memory usage normal: {total_memories} memories',
                        'details': {'total_memories': total_memories}
                    }
            
            health_checker.register_health_check("memory_usage", "system", custom_memory_check)
            
            # Add some memories to trigger the custom check
            for i in range(5):
                context_manager.memory_manager.add_memory(
                    content=f"Test memory {i}",
                    importance=0.5
                )
            
            # Perform health check with custom check
            health_status = await health_checker.perform_health_check()
            custom_checks = [check for check in health_status.checks if check.name == "memory_usage"]
            
            if custom_checks:
                print(f"Custom check result: {custom_checks[0].status}")
                print(f"Custom check message: {custom_checks[0].message}")
            
        finally:
            await health_checker.stop()
            await context_manager.cleanup()
    
    print("=== Health Checking Example Complete ===")


async def error_handling_example():
    """Demonstrate error handling and recovery features."""
    print("\n=== Error Handling Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=1000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Example 1: Invalid message handling
            print("\n1. Invalid message handling")
            print("-" * 40)
            
            try:
                await context_manager.add_message("", "Empty role should fail")
            except Exception as e:
                print(f"Caught expected error: {type(e).__name__}: {e}")
            
            # Example 2: Memory error handling
            print("\n2. Memory error handling")
            print("-" * 40)
            
            try:
                context_manager.memory_manager.add_memory("", importance=1.0)
            except Exception as e:
                print(f"Caught memory error: {type(e).__name__}: {e}")
            
            # Example 3: Tool execution error handling
            print("\n3. Tool execution error handling")
            print("-" * 40)
            
            # Register a faulty tool
            async def faulty_tool(parameters):
                raise ValueError("This tool always fails!")
            
            context_manager.tool_manager.register_function_tool("faulty_tool", faulty_tool, "A tool that always fails")
            
            try:
                result = await context_manager.execute_tool("faulty_tool")
                print(f"Tool execution result: {result.success}")
                if not result.success:
                    print(f"Error captured: {result.error}")
            except Exception as e:
                print(f"Caught tool error: {type(e).__name__}: {e}")
            
            # Example 4: Recovery mechanisms
            print("\n4. System recovery")
            print("-" * 40)
            
            # Simulate system stress
            print("Simulating system stress...")
            for i in range(10):
                try:
                    await context_manager.add_message("user", f"Stress test message {i}")
                    await context_manager.search_memory(f"stress {i}")
                except Exception as e:
                    print(f"Recovery from error: {e}")
            
            # Check system status after stress
            status = context_manager.get_system_status()
            print(f"System status after stress: {status['status']}")
            # Calculate simple health score from performance metrics
            perf = status['performance']
            health_score = 100.0
            if perf.get('errors', 0) > 0:
                health_score -= min(30, perf['errors'] * 10)
            if perf.get('memory_usage', 0) > 1000000:  # > 1MB
                health_score -= min(20, (perf['memory_usage'] - 1000000) / 100000)
            health_score = max(0, min(100, health_score))
            print(f"Performance health: {health_score:.2f}")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Error Handling Example Complete ===")


async def main():
    """Run all advanced feature examples."""
    print("Python Context Manager - Advanced Features Examples")
    print("=" * 60)
    
    await memory_management_example()
    await context_compression_example()
    await performance_monitoring_example()
    await concurrent_operations_example()
    await health_checking_example()
    await error_handling_example()
    
    print("\n" + "=" * 60)
    print("All advanced feature examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())