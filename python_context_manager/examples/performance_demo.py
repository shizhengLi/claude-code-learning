"""
Performance demonstration examples for the Python Context Manager.

This example demonstrates performance testing and benchmarking:
- Performance measurement and monitoring
- Load testing and stress testing
- Memory usage analysis
- Optimization techniques
- Benchmarking different configurations
- Performance comparison between features
"""

import asyncio
import tempfile
import time
import tracemalloc
import psutil
import gc
from typing import List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig
from context_manager.core.performance_monitor import PerformanceMonitor
from context_manager.core.async_operations import ConcurrencyController


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.results = {}
        
    async def measure_operation(self, name: str, operation, *args, **kwargs) -> Dict[str, Any]:
        """Measure the performance of an operation."""
        # Start memory tracking
        tracemalloc.start()
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_time = time.time()
        
        # Execute operation
        try:
            result = await operation(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Get final measurements
        final_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Get memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        execution_time = final_time - initial_time
        memory_delta = final_memory - initial_memory
        
        benchmark_result = {
            "name": name,
            "execution_time": execution_time,
            "memory_usage": final_memory,
            "memory_delta": memory_delta,
            "peak_memory": peak / 1024 / 1024,  # Convert to MB
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results[name] = benchmark_result
        return benchmark_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.results:
            return {"message": "No benchmarks run"}
        
        total_time = sum(r["execution_time"] for r in self.results.values())
        avg_time = total_time / len(self.results)
        max_time = max(r["execution_time"] for r in self.results.values())
        min_time = min(r["execution_time"] for r in self.results.values())
        
        total_memory = sum(r["memory_usage"] for r in self.results.values())
        avg_memory = total_memory / len(self.results)
        max_memory = max(r["memory_usage"] for r in self.results.values())
        
        success_rate = sum(1 for r in self.results.values() if r["success"]) / len(self.results)
        
        return {
            "total_benchmarks": len(self.results),
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "average_memory_usage": avg_memory,
            "max_memory_usage": max_memory,
            "success_rate": success_rate,
            "results": self.results
        }


async def basic_performance_benchmark():
    """Basic performance benchmarking of core operations."""
    print("=== Basic Performance Benchmark ===")
    
    benchmark = PerformanceBenchmark()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir,
            compression_ratio=0.8
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Benchmark 1: Initialization
            print("\n1. Initialization benchmark")
            print("-" * 40)
            
            async def init_benchmark():
                new_config = ContextManagerConfig(
                    max_tokens=1000,
                    cache_path=temp_dir + "/benchmark"
                )
                cm = EnhancedContextManager(config=new_config)
                await cm.initialize()
                await cm.cleanup()
                return True
            
            result = await benchmark.measure_operation("initialization", init_benchmark)
            print(f"Initialization time: {result['execution_time']:.3f}s")
            print(f"Memory usage: {result['memory_usage']:.2f}MB")
            
            # Benchmark 2: Message addition
            print("\n2. Message addition benchmark")
            print("-" * 40)
            
            async def add_messages_benchmark():
                for i in range(100):
                    await context_manager.add_message("user", f"Benchmark message {i}")
                    await context_manager.add_message("assistant", f"Benchmark response {i}")
                return True
            
            result = await benchmark.measure_operation("add_100_messages", add_messages_benchmark)
            print(f"Added 200 messages in {result['execution_time']:.3f}s")
            print(f"Average per message: {result['execution_time']/200:.4f}s")
            print(f"Memory delta: {result['memory_delta']:+.2f}MB")
            
            # Benchmark 3: Memory search
            print("\n3. Memory search benchmark")
            print("-" * 40)
            
            async def search_benchmark():
                for i in range(50):
                    await context_manager.search_memory(f"benchmark {i % 10}")
                return True
            
            result = await benchmark.measure_operation("search_50_times", search_benchmark)
            print(f"50 searches in {result['execution_time']:.3f}s")
            print(f"Average per search: {result['execution_time']/50:.4f}s")
            
            # Benchmark 4: Context compression
            print("\n4. Context compression benchmark")
            print("-" * 40)
            
            # Add many messages to trigger compression
            for i in range(50):
                await context_manager.add_message("user", f"Long message for compression testing {i}: " + "This is a very long message that will trigger the context compression mechanism when the token limit is reached. " * 5)
                await context_manager.add_message("assistant", f"Long response for compression testing {i}: " + "This is a very long response that will also contribute to the context compression mechanism. " * 5)
            
            # Clear previous compression history
            context_manager.compression_history.clear()
            
            async def compression_benchmark():
                # Add more messages to trigger compression
                for i in range(20):
                    await context_manager.add_message("user", f"Trigger compression {i}: " + "Very long message content " * 10)
                    await context_manager.add_message("assistant", f"Trigger compression response {i}: " + "Very long response content " * 10)
                return len(context_manager.compression_history) > 0
            
            result = await benchmark.measure_operation("context_compression", compression_benchmark)
            print(f"Compression operations in {result['execution_time']:.3f}s")
            if context_manager.compression_history:
                avg_compression = sum(c['compression_ratio'] for c in context_manager.compression_history) / len(context_manager.compression_history)
                print(f"Average compression ratio: {avg_compression:.2%}")
            
        finally:
            await context_manager.cleanup()
    
    # Print summary
    summary = benchmark.get_summary()
    print("\n5. Benchmark summary")
    print("-" * 40)
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Average execution time: {summary['average_execution_time']:.3f}s")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    print("=== Basic Performance Benchmark Complete ===")


async def load_testing_example():
    """Load testing with concurrent operations."""
    print("\n=== Load Testing Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=5000,
            cache_path=temp_dir,
            compression_ratio=0.7
        )
        
        context_manager = EnhancedContextManager(config=config)
        concurrency_controller = ConcurrencyController(max_concurrent_tasks=20)
        
        await context_manager.initialize()
        await concurrency_controller.start()
        
        try:
            # Setup performance monitoring
            performance_monitor = PerformanceMonitor(sample_interval=0.1)
            await performance_monitor.start_monitoring()
            
            # Define load test scenarios
            async def user_session(session_id: int, duration: float) -> Dict[str, Any]:
                """Simulate a user session."""
                start_time = time.time()
                operations = 0
                errors = 0
                
                while time.time() - start_time < duration:
                    try:
                        # Add message
                        await context_manager.add_message("user", f"Session {session_id} message {operations}")
                        operations += 1
                        
                        # Search memory
                        await context_manager.search_memory(f"session {session_id}")
                        operations += 1
                        
                        # Small delay
                        await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        errors += 1
                        print(f"Session {session_id} error: {e}")
                
                return {
                    "session_id": session_id,
                    "operations": operations,
                    "errors": errors,
                    "duration": duration,
                    "operations_per_second": operations / duration
                }
            
            # Run load test with different concurrency levels
            concurrency_levels = [5, 10, 20]
            
            for concurrency in concurrency_levels:
                print(f"\n1. Load test with {concurrency} concurrent users")
                print("-" * 50)
                
                # Start concurrent sessions
                task_ids = []
                start_time = time.time()
                
                for i in range(concurrency):
                    task_id = await concurrency_controller.submit_task(
                        f"load_test_session_{i}",
                        user_session,
                        i,
                        5.0,  # 5 seconds duration
                        priority="normal"
                    )
                    task_ids.append(task_id)
                
                # Wait for all sessions to complete
                results = []
                for task_id in task_ids:
                    result = await concurrency_controller.wait_for_task(task_id)
                    results.append(result)
                
                total_time = time.time() - start_time
                
                # Analyze results
                total_operations = sum(r["operations"] for r in results)
                total_errors = sum(r["errors"] for r in results)
                avg_ops_per_second = sum(r["operations_per_second"] for r in results) / len(results)
                
                print(f"Total time: {total_time:.2f}s")
                print(f"Total operations: {total_operations}")
                print(f"Total errors: {total_errors}")
                print(f"Average operations per second: {avg_ops_per_second:.1f}")
                print(f"System throughput: {total_operations / total_time:.1f} ops/sec")
                
                # Get performance metrics
                metrics = performance_monitor.get_current_metrics()
                print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
                print(f"CPU usage: {metrics.cpu_usage_percent:.1f}%")
                
                # Check system health
                health_score = performance_monitor._calculate_health_score(metrics)
                print(f"System health: {health_score:.2f}")
                
                # Wait between tests
                await asyncio.sleep(1)
            
            # Stress test
            print("\n2. Stress test - Maximum load")
            print("-" * 50)
            
            # Push system to limits
            stress_tasks = []
            stress_start = time.time()
            
            for i in range(50):  # Very high load
                task_id = await concurrency_controller.submit_task(
                    f"stress_test_{i}",
                    user_session,
                    i,
                    2.0,  # Short duration
                    priority="normal"
                )
                stress_tasks.append(task_id)
            
            # Monitor during stress test
            while not all(concurrency_controller.get_task_status(tid) in ["completed", "failed"] for tid in stress_tasks):
                resource_usage = concurrency_controller.get_resource_usage()
                print(f"  Active: {resource_usage.active_tasks}, Queued: {resource_usage.queued_tasks}, Memory: {resource_usage.memory_usage_mb:.1f}MB")
                await asyncio.sleep(0.5)
            
            stress_time = time.time() - stress_start
            print(f"Stress test completed in {stress_time:.2f}s")
            
            # Get final performance metrics
            final_metrics = performance_monitor.get_current_metrics()
            print(f"Final memory usage: {final_metrics.memory_usage_mb:.2f}MB")
            print(f"Peak memory during test: {final_metrics.memory_usage_mb:.2f}MB")
            
            await performance_monitor.stop_monitoring()
            
        finally:
            await concurrency_controller.stop()
            await context_manager.cleanup()
    
    print("=== Load Testing Example Complete ===")


async def memory_usage_analysis():
    """Analyze memory usage patterns."""
    print("\n=== Memory Usage Analysis ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=3000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Start memory tracking
            tracemalloc.start()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"Initial memory usage: {initial_memory:.2f}MB")
            
            # Memory usage during different operations
            memory_snapshots = []
            
            # Phase 1: Add messages
            print("\n1. Memory usage during message addition")
            print("-" * 45)
            
            for batch in range(5):
                batch_start = time.time()
                
                for i in range(20):
                    await context_manager.add_message("user", f"Memory test message {batch}-{i}")
                    await context_manager.add_message("assistant", f"Memory test response {batch}-{i}")
                
                # Take memory snapshot
                current, peak = tracemalloc.get_traced_memory()
                current_memory = process.memory_info().rss / 1024 / 1024
                
                snapshot = {
                    "phase": f"batch_{batch}",
                    "messages_added": (batch + 1) * 40,
                    "current_memory": current_memory,
                    "peak_memory": peak / 1024 / 1024,
                    "memory_delta": current_memory - initial_memory,
                    "time_elapsed": time.time() - batch_start
                }
                memory_snapshots.append(snapshot)
                
                print(f"  Batch {batch}: {snapshot['messages_added']} messages, "
                      f"{snapshot['current_memory']:.2f}MB (+{snapshot['memory_delta']:.1f}MB)")
            
            # Phase 2: Memory search operations
            print("\n2. Memory usage during search operations")
            print("-" * 45)
            
            search_start_memory = process.memory_info().rss / 1024 / 1024
            
            for i in range(20):
                await context_manager.search_memory(f"memory test {i % 5}")
                
                if i % 5 == 4:  # Every 5 searches
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"  After {i+1} searches: {current_memory:.2f}MB "
                          f"(+{current_memory - search_start_memory:.1f}MB)")
            
            # Phase 3: Context compression
            print("\n3. Memory usage during context compression")
            print("-" * 45)
            
            compression_start_memory = process.memory_info().rss / 1024 / 1024
            
            # Add messages to trigger compression
            for i in range(30):
                await context_manager.add_message("user", f"Long compression test message {i}: " + "This is a very long message designed to test context compression and memory usage patterns under load. " * 3)
                await context_manager.add_message("assistant", f"Long compression test response {i}: " + "This is a very long response designed to test how the system handles memory during compression operations. " * 3)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"  Before compression: {compression_start_memory:.2f}MB")
            print(f"  After compression: {final_memory:.2f}MB")
            print(f"  Memory change: {final_memory - compression_start_memory:+.2f}MB")
            
            if context_manager.compression_history:
                total_compressed = sum(c['tokens_before'] - c['tokens_after'] for c in context_manager.compression_history)
                print(f"  Total tokens compressed: {total_compressed}")
            
            # Phase 4: Memory cleanup
            print("\n4. Memory cleanup analysis")
            print("-" * 45)
            
            # Force garbage collection
            gc.collect()
            cleanup_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"  Before GC: {final_memory:.2f}MB")
            print(f"  After GC: {cleanup_memory:.2f}MB")
            print(f"  Memory freed: {final_memory - cleanup_memory:.2f}MB")
            
            # Memory analysis summary
            print("\n5. Memory analysis summary")
            print("-" * 45)
            
            total_memory_change = cleanup_memory - initial_memory
            total_messages = len(context_manager.context_window.messages)
            
            print(f"Initial memory: {initial_memory:.2f}MB")
            print(f"Final memory: {cleanup_memory:.2f}MB")
            print(f"Total change: {total_memory_change:+.2f}MB")
            print(f"Total messages: {total_messages}")
            print(f"Memory per message: {total_memory_change / total_messages:.3f}MB" if total_messages > 0 else "N/A")
            
            # Stop memory tracking
            tracemalloc.stop()
            
        finally:
            await context_manager.cleanup()
    
    print("=== Memory Usage Analysis Complete ===")


async def optimization_comparison():
    """Compare different optimization strategies."""
    print("\n=== Optimization Comparison ===")
    
    # Test different configurations
    configurations = {
        "lightweight": ContextManagerConfig(
            max_tokens=1000,
            compression_ratio=0.9,
            short_term_memory_size=500
        ),
        "balanced": ContextManagerConfig(
            max_tokens=2000,
            compression_ratio=0.8,
            short_term_memory_size=1000,
            medium_term_memory_size=2000
        ),
        "heavy": ContextManagerConfig(
            max_tokens=5000,
            compression_ratio=0.6,
            short_term_memory_size=2000,
            medium_term_memory_size=5000,
            long_term_memory_size=10000
        )
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\n1. Testing {config_name} configuration")
        print("-" * 45)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.cache_path = temp_dir + f"/{config_name}"
            
            context_manager = EnhancedContextManager(config=config)
            await context_manager.initialize()
            
            try:
                benchmark = PerformanceBenchmark()
                performance_monitor = PerformanceMonitor()
                await performance_monitor.start_monitoring()
                
                # Standard test suite
                async def standard_test_suite():
                    # Add messages
                    for i in range(50):
                        await context_manager.add_message("user", f"Test message {i}")
                        await context_manager.add_message("assistant", f"Test response {i}")
                    
                    # Search operations
                    for i in range(20):
                        await context_manager.search_memory(f"test {i % 10}")
                    
                    # Memory operations
                    for i in range(10):
                        await context_manager.memory_manager.add_memory(
                            content=f"Test memory {i}",
                            importance=0.5
                        )
                    
                    return True
                
                # Run benchmark
                result = await benchmark.measure_operation(f"{config_name}_test_suite", standard_test_suite)
                
                # Get system metrics
                status = context_manager.get_system_status()
                summary = context_manager.get_context_summary()
                metrics = performance_monitor.get_current_metrics()
                health_score = performance_monitor._calculate_health_score(metrics)
                
                results[config_name] = {
                    "benchmark": result,
                    "system_status": status,
                    "context_summary": summary,
                    "health_score": health_score,
                    "config": {
                        "max_tokens": config.max_tokens,
                        "compression_ratio": config.compression_ratio,
                        "memory_sizes": {
                            "short_term": config.short_term_memory_size,
                            "medium_term": config.medium_term_memory_size,
                            "long_term": config.long_term_memory_size
                        }
                    }
                }
                
                print(f"  Execution time: {result['execution_time']:.3f}s")
                print(f"  Memory usage: {result['memory_usage']:.2f}MB")
                print(f"  System health: {health_score:.2f}")
                print(f"  Messages processed: {summary['message_count']}")
                print(f"  Context utilization: {summary['utilization']:.2%}")
                
            finally:
                await performance_monitor.stop_monitoring()
                await context_manager.cleanup()
    
    # Compare results
    print("\n2. Configuration comparison")
    print("-" * 45)
    
    print(f"{'Configuration':<12} {'Time (s)':<8} {'Memory (MB)':<12} {'Health':<8} {'Messages':<10} {'Utilization':<12}")
    print("-" * 70)
    
    for config_name, result in results.items():
        benchmark = result['benchmark']
        health_score = result['health_score']
        summary = result['context_summary']
        
        print(f"{config_name:<12} {benchmark['execution_time']:<8.3f} {benchmark['memory_usage']:<12.2f} "
              f"{health_score:<8.2f} {summary['message_count']:<10} "
              f"{summary['utilization']:<12.1%}")
    
    # Find best configuration for different metrics
    print("\n3. Performance analysis")
    print("-" * 45)
    
    # Fastest configuration
    fastest = min(results.items(), key=lambda x: x[1]['benchmark']['execution_time'])
    print(f"Fastest configuration: {fastest[0]} ({fastest[1]['benchmark']['execution_time']:.3f}s)")
    
    # Most memory efficient
    most_efficient = min(results.items(), key=lambda x: x[1]['benchmark']['memory_usage'])
    print(f"Most memory efficient: {most_efficient[0]} ({most_efficient[1]['benchmark']['memory_usage']:.2f}MB)")
    
    # Highest health score
    healthiest = max(results.items(), key=lambda x: x[1]['health_score'])
    print(f"Healthiest configuration: {healthiest[0]} (health: {healthiest[1]['health_score']:.2f})")
    
    print("=== Optimization Comparison Complete ===")


async def performance_monitoring_demo():
    """Demonstrate performance monitoring capabilities."""
    print("\n=== Performance Monitoring Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        performance_monitor = PerformanceMonitor(sample_interval=0.05)  # Fast sampling
        
        await context_manager.initialize()
        await performance_monitor.start_monitoring()
        
        try:
            print("\n1. Running mixed workload")
            print("-" * 40)
            
            # Mixed workload with different operation types
            workload = []
            
            # Message operations
            for i in range(20):
                workload.append(("message", f"Workload message {i}"))
            
            # Search operations
            for i in range(15):
                workload.append(("search", f"workload search {i % 5}"))
            
            # Memory operations
            for i in range(10):
                workload.append(("memory", f"workload memory {i}"))
            
            # Execute workload with monitoring
            start_time = time.time()
            
            for operation_type, content in workload:
                op_start = time.time()
                
                if operation_type == "message":
                    await context_manager.add_message("user", content)
                    await context_manager.add_message("assistant", f"Response to {content}")
                elif operation_type == "search":
                    await context_manager.search_memory(content)
                elif operation_type == "memory":
                    await context_manager.memory_manager.add_memory(content, importance=0.5)
                
                op_duration = time.time() - op_start
                performance_monitor.record_operation(
                    f"{operation_type}_operation",
                    op_duration,
                    True
                )
                
                # Simulate cache operations
                if operation_type == "search":
                    performance_monitor.record_cache_operation(hit=True)
                else:
                    performance_monitor.record_cache_operation(hit=False)
            
            workload_time = time.time() - start_time
            
            print(f"Workload completed in {workload_time:.2f}s")
            print(f"Operations per second: {len(workload) / workload_time:.1f}")
            
            # Get performance metrics
            print("\n2. Real-time performance metrics")
            print("-" * 40)
            
            metrics = performance_monitor.get_current_metrics()
            print(f"Message operations: {metrics.message_operations}")
            print(f"Memory operations: {metrics.memory_operations}")
            print(f"Cache hits: {metrics.cache_hits}")
            print(f"Cache misses: {metrics.cache_misses}")
            print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
            print(f"Error count: {metrics.error_count}")
            print(f"Health score: {metrics.health_score:.2f}")
            print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
            print(f"CPU usage: {metrics.cpu_usage_percent:.1f}%")
            
            # Performance trends
            print("\n3. Performance trends")
            print("-" * 40)
            
            trends = performance_monitor.get_performance_trends()
            
            if 'response_time_trend' in trends:
                print(f"Response time trend: {trends['response_time_trend']}")
            if 'error_rate_trend' in trends:
                print(f"Error rate trend: {trends['error_rate_trend']}")
            if 'cache_efficiency_trend' in trends:
                print(f"Cache efficiency trend: {trends['cache_efficiency_trend']}")
            
            # Performance summary
            print("\n4. Performance summary")
            print("-" * 40)
            
            summary = performance_monitor.get_performance_summary()
            print(f"Overall health score: {summary['health_score']:.2f}")
            print(f"Total operations: {summary['total_operations']}")
            print(f"Average response time: {summary['average_response_time']:.3f}s")
            print(f"Peak memory usage: {summary['peak_memory_usage']:.2f}MB")
            
            if summary['recommendations']:
                print("Recommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
            
            # Performance alerts
            print("\n5. Performance alerts")
            print("-" * 40)
            
            alerts = []
            if metrics.health_score < 0.7:
                alerts.append("Low health score detected")
            if metrics.memory_usage_mb > 100:
                alerts.append("High memory usage detected")
            if metrics.cache_hit_rate < 0.5:
                alerts.append("Low cache efficiency detected")
            
            if alerts:
                for alert in alerts:
                    print(f"  ⚠️  {alert}")
            else:
                print("  ✅ No performance alerts")
            
        finally:
            await performance_monitor.stop_monitoring()
            await context_manager.cleanup()
    
    print("=== Performance Monitoring Demo Complete ===")


async def main():
    """Run all performance demonstration examples."""
    print("Python Context Manager - Performance Demonstration")
    print("=" * 60)
    
    await basic_performance_benchmark()
    await load_testing_example()
    await memory_usage_analysis()
    await optimization_comparison()
    await performance_monitoring_demo()
    
    print("\n" + "=" * 60)
    print("All performance demonstrations completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())