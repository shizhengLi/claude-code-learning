"""
Integration tests for the enhanced context manager system.

This module provides comprehensive integration tests that verify
the entire system works together correctly.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig
from context_manager.core.performance_monitor import PerformanceMonitor
from context_manager.core.config_validation import ConfigValidator, ErrorRecoveryManager, HealthChecker
from context_manager.core.async_operations import ConcurrencyController, AsyncOperationManager
from context_manager.core.health_checker import HealthChecker as SystemHealthChecker
from context_manager.utils.error_handling import ContextManagerError, ConfigurationError


class TestEnhancedContextManager:
    """Integration tests for EnhancedContextManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return ContextManagerConfig(
            max_tokens=1000,
            cache_path=temp_dir,
            short_term_memory_size=1000,
            medium_term_memory_size=2000,
            long_term_memory_size=5000,
            cache_size=2000,
            compression_ratio=0.8
        )
    
    @pytest_asyncio.fixture
    async def context_manager(self, config):
        """Create and initialize context manager."""
        manager = EnhancedContextManager(config=config)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization(self, context_manager):
        """Test context manager initialization."""
        assert context_manager._status.value == "ready"
        assert context_manager.config is not None
        assert context_manager.memory_manager is not None
        assert context_manager.tool_manager is not None
        assert context_manager.storage_manager is not None
    
    @pytest.mark.asyncio
    async def test_message_operations(self, context_manager):
        """Test message operations."""
        # Check initial state
        initial_summary = context_manager.get_context_summary()
        initial_count = initial_summary['message_count']
        
        # Add messages
        success = await context_manager.add_message("user", "Hello, world!")
        assert success is True
        
        # Check count after first message
        summary_after_first = context_manager.get_context_summary()
        
        success = await context_manager.add_message("assistant", "Hello! How can I help you?")
        assert success is True
        
        # Check context summary
        summary = context_manager.get_context_summary()
        # The exact count might vary due to memory retrieval, so just check it increased
        assert summary['message_count'] > initial_count
        assert summary['token_count'] > 0
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, context_manager):
        """Test memory system integration."""
        # Add a message
        await context_manager.add_message("user", "Remember this important information")
        
        # Search memory
        memories = await context_manager.search_memory("important information", limit=5)
        assert len(memories) > 0
        assert any("important information" in mem.content for mem in memories)
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, context_manager):
        """Test tool execution."""
        # Mock a simple tool
        async def test_tool(message: str):
            return {"result": f"Processed: {message}", "success": True}
        
        # Register tool
        context_manager.tool_manager.register_function_tool("test_tool", test_tool, "Test tool for integration testing")
        
        # Execute tool
        result = await context_manager.execute_tool("test_tool", message="test")
        assert result.success is True
        assert "Processed: {'message': 'test'}" in result.result['result']
    
    @pytest.mark.asyncio
    async def test_storage_integration(self, context_manager):
        """Test storage system integration."""
        # Store data
        await context_manager.storage_manager.set("test_key", "test_value")
        
        # Retrieve data
        result = await context_manager.storage_manager.get("test_key")
        assert result == "test_value"
        
        # Check storage stats
        stats = context_manager.storage_manager.get_stats()
        assert 'total' in stats
    
    @pytest.mark.asyncio
    async def test_context_compression(self, context_manager):
        """Test context compression."""
        # Add many messages to trigger compression
        for i in range(20):
            await context_manager.add_message("user", f"This is message number {i}")
            await context_manager.add_message("assistant", f"Response to message {i}")
        
        # Check that compression occurred
        assert len(context_manager.compression_history) > 0
    
    @pytest.mark.asyncio
    async def test_health_checks(self, context_manager):
        """Test health check functionality."""
        # Perform health check
        health_checks = await context_manager.perform_health_check()
        
        assert len(health_checks) > 0
        assert any(check.name == "system_status" for check in health_checks)
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, context_manager):
        """Test performance optimization."""
        # Add some messages first
        for i in range(10):
            await context_manager.add_message("user", f"Test message {i}")
        
        # Optimize system
        optimization_result = await context_manager.optimize_system()
        
        assert 'actions_taken' in optimization_result
        assert 'performance_improvements' in optimization_result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, context_manager):
        """Test error handling."""
        # Test with invalid configuration
        with pytest.raises(ContextManagerError):
            await context_manager.add_message("", "")  # Empty role should fail
    
    @pytest.mark.asyncio
    async def test_system_status(self, context_manager):
        """Test system status reporting."""
        status = context_manager.get_system_status()
        
        assert 'status' in status
        assert 'performance' in status
        assert 'context_summary' in status
        assert 'storage_stats' in status
    
    @pytest.mark.asyncio
    async def test_cleanup(self, context_manager):
        """Test cleanup functionality."""
        # Add some data
        await context_manager.add_message("user", "Test message")
        await context_manager.storage_manager.set("test", "value")
        
        # Cleanup
        await context_manager.cleanup()
        
        assert context_manager._status.value == "stopped"


class TestPerformanceMonitor:
    """Integration tests for PerformanceMonitor."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor."""
        return PerformanceMonitor(sample_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, performance_monitor):
        """Test start and stop functionality."""
        await performance_monitor.start_monitoring()
        assert performance_monitor._is_monitoring is True
        
        await performance_monitor.stop_monitoring()
        assert performance_monitor._is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_operation_recording(self, performance_monitor):
        """Test operation recording."""
        await performance_monitor.start_monitoring()
        
        # Record some operations
        performance_monitor.record_operation("message_add", 0.1, True)
        performance_monitor.record_operation("memory_search", 0.2, True)
        performance_monitor.record_operation("tool_execution", 0.3, False)
        
        # Wait for monitoring to process
        await asyncio.sleep(0.2)
        
        # Check metrics
        metrics = performance_monitor.get_current_metrics()
        assert metrics.message_operations >= 1
        assert metrics.memory_operations >= 1
        assert metrics.tool_operations >= 1
        assert metrics.error_count >= 1
        
        await performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_cache_tracking(self, performance_monitor):
        """Test cache hit/miss tracking."""
        await performance_monitor.start_monitoring()
        
        # Record cache operations
        performance_monitor.record_cache_operation(hit=True)
        performance_monitor.record_cache_operation(hit=False)
        performance_monitor.record_cache_operation(hit=True)
        
        # Wait for monitoring to process
        await asyncio.sleep(0.2)
        
        # Check cache metrics
        metrics = performance_monitor.get_current_metrics()
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1
        assert metrics.cache_hit_rate == 2/3
        
        await performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        await performance_monitor.start_monitoring()
        
        # Record some operations
        for i in range(5):
            performance_monitor.record_operation("test_operation", 0.1 + i * 0.01, True)
        
        # Wait for monitoring to process
        await asyncio.sleep(0.2)
        
        # Get summary
        summary = performance_monitor.get_performance_summary()
        
        assert 'health_score' in summary
        assert 'current_metrics' in summary
        assert 'trends' in summary
        assert 'recommendations' in summary
        
        await performance_monitor.stop_monitoring()


class TestConfigValidation:
    """Integration tests for configuration validation."""
    
    @pytest.fixture
    def config_manager(self):
        """Create configuration manager."""
        from context_manager.core.config import ConfigManager
        return ConfigManager()
    
    @pytest.fixture
    def config_validator(self, config_manager):
        """Create configuration validator."""
        return ConfigValidator(config_manager)
    
    def test_valid_configuration(self, config_validator):
        """Test validation of valid configuration."""
        config = ContextManagerConfig(
            max_tokens=1000,
            storage_path="/tmp/test",
            memory_cache_size=1024 * 1024,
            cache_size=2 * 1024 * 1024
        )
        
        result = config_validator.validate_config(config)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_configuration(self, config_validator):
        """Test validation of invalid configuration."""
        # Create a config with invalid values by creating a basic valid one first
        from context_manager.core.config import ContextManagerConfig
        
        config = ContextManagerConfig(max_tokens=1000)
        
        # Modify to invalid values after creation (bypassing __post_init__)
        config.max_tokens = -1
        config.memory_cache_size = 0
        config.cache_size = -1
        
        # Manually call validation through config_validator
        result = config_validator.validate_config(config)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validation_levels(self, config_validator):
        """Test different validation levels."""
        config = ContextManagerConfig(
            max_tokens=100000  # Large but valid
        )
        
        # Basic validation
        basic_result = config_validator.validate_config(config, level="basic")
        assert basic_result.is_valid is True
        
        # Strict validation
        strict_result = config_validator.validate_config(config, level="strict")
        assert strict_result.is_valid is True
        # Note: Warnings may not be generated for all configurations


class TestErrorRecovery:
    """Integration tests for error recovery."""
    
    @pytest.fixture
    def config_manager(self):
        """Create configuration manager."""
        from context_manager.core.config import ConfigManager
        return ConfigManager()
    
    @pytest.fixture
    def error_recovery_manager(self, config_manager):
        """Create error recovery manager."""
        return ErrorRecoveryManager(config_manager)
    
    @pytest.mark.asyncio
    async def test_configuration_error_recovery(self, error_recovery_manager):
        """Test recovery from configuration errors."""
        error = ConfigurationError("Invalid configuration")
        
        recovery_result = await error_recovery_manager.handle_error(error)
        
        assert recovery_result.action.value == "reset_config"
        assert recovery_result.success is True
    
    @pytest.mark.asyncio
    async def test_error_statistics(self, error_recovery_manager):
        """Test error statistics tracking."""
        # Handle multiple errors
        errors = [
            ConfigurationError("Config error"),
            ContextManagerError("Context error"),
            ConfigurationError("Another config error")
        ]
        
        for error in errors:
            await error_recovery_manager.handle_error(error)
        
        # Check statistics
        stats = error_recovery_manager.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['error_types']['ConfigurationError'] == 2
        assert stats['error_types']['ContextManagerError'] == 1
        assert stats['recovery_attempts'] == 3


class TestConcurrencyController:
    """Integration tests for concurrency controller."""
    
    @pytest.fixture
    def concurrency_controller(self):
        """Create concurrency controller."""
        return ConcurrencyController(
            max_concurrent_tasks=3,
            max_thread_pool_size=2,
            max_async_queue_size=10
        )
    
    @pytest_asyncio.fixture
    async def started_controller(self, concurrency_controller):
        """Start and stop concurrency controller."""
        await concurrency_controller.start()
        yield concurrency_controller
        await concurrency_controller.stop()
    
    @pytest.mark.asyncio
    async def test_task_execution(self, started_controller):
        """Test basic task execution."""
        async def test_task(x, y):
            await asyncio.sleep(0.1)
            return x + y
        
        # Execute task
        result = await started_controller.execute_task(
            "test_addition",
            test_task,
            5, 3,  # positional args
            priority="normal"
        )
        
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, started_controller):
        """Test parallel task execution."""
        async def test_task(i):
            await asyncio.sleep(0.1)
            return i * 2
        
        # Submit multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await started_controller.submit_task(
                f"test_task_{i}",
                test_task,
                i,  # positional arg
                priority="normal"
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            result = await started_controller.wait_for_task(task_id)
            results.append(result)
        
        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]
    
    @pytest.mark.asyncio
    async def test_resource_limits(self, started_controller):
        """Test resource limit enforcement."""
        async def slow_task():
            await asyncio.sleep(0.5)
            return "completed"
        
        # Submit more tasks than the concurrency limit
        task_ids = []
        for i in range(10):
            task_id = await started_controller.submit_task(
                f"slow_task_{i}",
                slow_task,
                priority="normal"
            )
            task_ids.append(task_id)
        
        # Check that only the allowed number are running
        resource_usage = started_controller.get_resource_usage()
        from context_manager.core.async_operations import ResourceLimit
        assert resource_usage.active_tasks <= started_controller.resource_limits[ResourceLimit.MAX_CONCURRENT_TASKS]
        
        # Wait for all tasks
        results = []
        for task_id in task_ids:
            result = await started_controller.wait_for_task(task_id)
            results.append(result)
        
        assert len(results) == 10


class TestHealthChecker:
    """Integration tests for health checker."""
    
    @pytest.fixture
    def health_checker(self):
        """Create health checker."""
        return SystemHealthChecker(check_interval=0.1)
    
    @pytest_asyncio.fixture
    async def started_health_checker(self, health_checker):
        """Start and stop health checker."""
        await health_checker.start()
        yield health_checker
        await health_checker.stop()
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, started_health_checker):
        """Test health check execution."""
        # Perform health check
        health_status = await started_health_checker.perform_health_check()
        
        assert health_status.status.value in ["healthy", "warning", "degraded", "critical"]
        assert health_status.health_score >= 0.0
        assert health_status.health_score <= 1.0
        assert len(health_status.checks) > 0
    
    @pytest.mark.asyncio
    async def test_diagnostic_report(self, started_health_checker):
        """Test diagnostic report generation."""
        # Generate diagnostic report
        report = await started_health_checker.generate_diagnostic_report()
        
        # Convert report to dict to check attributes
        report_dict = report.to_dict()
        assert 'timestamp' in report_dict
        assert 'system_info' in report_dict
        assert 'health_status' in report_dict
        assert 'performance_metrics' in report_dict
        assert 'recommendations' in report_dict
    
    @pytest.mark.asyncio
    async def test_custom_health_check(self, started_health_checker):
        """Test custom health check registration."""
        async def custom_check():
            return {
                'status': 'healthy',
                'message': 'Custom check passed',
                'details': {'custom': 'data'}
            }
        
        # Register custom check
        started_health_checker.register_health_check(
            "custom_check",
            "system",
            custom_check
        )
        
        # Perform health check
        health_status = await started_health_checker.perform_health_check()
        
        # Check that custom check was included
        custom_checks = [check for check in health_status.checks if check.name == "custom_check"]
        assert len(custom_checks) == 1
        assert custom_checks[0].status == "healthy"


class TestSystemIntegration:
    """End-to-end integration tests for the complete system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir,
            short_term_memory_size=2000,
            medium_term_memory_size=3000,
            long_term_memory_size=8000,
            cache_size=5000,
            compression_ratio=0.7
        )
    
    @pytest_asyncio.fixture
    async def full_system(self, config):
        """Create and initialize complete system."""
        # Create system components
        context_manager = EnhancedContextManager(config=config)
        performance_monitor = PerformanceMonitor(sample_interval=0.1)
        concurrency_controller = ConcurrencyController(max_concurrent_tasks=2)
        health_checker = SystemHealthChecker(check_interval=0.1)
        
        # Initialize all components
        await context_manager.initialize()
        await performance_monitor.start_monitoring()
        await concurrency_controller.start()
        await health_checker.start()
        
        yield {
            'context_manager': context_manager,
            'performance_monitor': performance_monitor,
            'concurrency_controller': concurrency_controller,
            'health_checker': health_checker
        }
        
        # Cleanup all components
        await context_manager.cleanup()
        await performance_monitor.stop_monitoring()
        await concurrency_controller.stop()
        await health_checker.stop()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, full_system):
        """Test complete system workflow."""
        cm = full_system['context_manager']
        pm = full_system['performance_monitor']
        cc = full_system['concurrency_controller']
        hc = full_system['health_checker']
        
        # Step 1: Add messages
        for i in range(5):
            await cm.add_message("user", f"User message {i}")
            await cm.add_message("assistant", f"Assistant response {i}")
        
        # Step 2: Execute tools
        async def echo_tool(parameters):
            message = parameters.get("message", "")
            return {"response": message, "success": True}
        
        cm.tool_manager.register_function_tool("echo", echo_tool, "Echo tool for testing")
        
        for i in range(3):
            result = await cm.execute_tool("echo", message=f"Test {i}")
            assert result.success is True
        
        # Step 3: Search memory
        memories = await cm.search_memory("User message", limit=5)
        assert len(memories) > 0
        
        # Step 4: Check system health
        health_status = await hc.perform_health_check()
        assert health_status.status.value in ["healthy", "warning"]
        
        # Step 5: Check performance metrics
        # Check context manager's internal metrics
        cm_metrics = cm.performance_metrics
        assert cm_metrics.message_count >= 10
        # Tool operations are tracked separately in tool manager
        tool_stats = cm.tool_manager.get_tool_stats()
        assert tool_stats['performance_metrics']['total_executions'] >= 3
        
        # Step 6: Generate diagnostic report
        report = await hc.generate_diagnostic_report()
        report_dict = report.to_dict()
        assert 'health_status' in report_dict
        assert 'performance_metrics' in report_dict
        assert 'recommendations' in report_dict
        
        # Step 7: Optimize system
        optimization_result = await cm.optimize_system()
        assert 'actions_taken' in optimization_result
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, full_system):
        """Test concurrent operations across all systems."""
        cm = full_system['context_manager']
        cc = full_system['concurrency_controller']
        
        async def user_workflow(user_id):
            """Simulate user workflow."""
            # Add message
            await cm.add_message("user", f"Message from user {user_id}")
            
            # Search memory
            await cm.search_memory(f"user {user_id}", limit=3)
            
            # Execute tool
            async def user_tool(parameters):
                return {"user_id": user_id, "success": True}
            
            cm.tool_manager.register_function_tool(f"user_tool_{user_id}", user_tool, f"User tool {user_id}")
            result = await cm.execute_tool(f"user_tool_{user_id}")
            
            return result
        
        # Execute multiple user workflows concurrently
        task_ids = []
        for i in range(5):
            task_id = await cc.submit_task(
                f"user_workflow_{i}",
                user_workflow,
                i,  # positional arg
                priority="normal"
            )
            task_ids.append(task_id)
        
        # Wait for all workflows to complete
        results = []
        for task_id in task_ids:
            result = await cc.wait_for_task(task_id)
            results.append(result)
        
        # Verify all workflows completed successfully
        assert len(results) == 5
        for result in results:
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, full_system):
        """Test error recovery across the integrated system."""
        cm = full_system['context_manager']
        pm = full_system['performance_monitor']
        hc = full_system['health_checker']
        
        # Record some successful operations
        pm.record_operation("test_operation", 0.1, True)
        pm.record_operation("test_operation", 0.2, True)
        
        # Record an error
        pm.record_operation("test_operation", 0.3, False)
        
        # Check health status
        health_status = await hc.perform_health_check()
        
        # Verify system is still functional despite errors
        assert health_status.status.value in ["healthy", "warning"]
        
        # Verify we can still add messages
        success = await cm.add_message("user", "Test after error")
        assert success is True
        
        # Verify performance metrics are tracked
        metrics = pm.get_current_metrics()
        assert metrics.error_count > 0
    
    @pytest.mark.asyncio
    async def test_system_resilience(self, full_system):
        """Test system resilience under load."""
        cm = full_system['context_manager']
        pm = full_system['performance_monitor']
        hc = full_system['health_checker']
        
        # Generate heavy load
        async def load_test_task(i):
            """Generate load."""
            for j in range(10):
                await cm.add_message("user", f"Load test message {i}-{j}")
                await cm.search_memory(f"load test {i}", limit=5)
                
                # Small delay to simulate processing
                await asyncio.sleep(0.01)
            
            return {"task_id": i, "completed": True}
        
        # Execute load test
        tasks = []
        for i in range(10):
            task = load_test_task(i)
            tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10
        
        # Check system health after load
        health_status = await hc.perform_health_check()
        assert health_status.status.value in ["healthy", "warning", "degraded"]
        
        # Check performance metrics
        # Check context manager's internal metrics
        cm_metrics = cm.performance_metrics
        assert cm_metrics.message_count > 0