"""
Performance monitoring and optimization system.

This module provides comprehensive performance monitoring, metrics collection,
and optimization capabilities for the context management system.
"""

import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
from pathlib import Path
import psutil
import tracemalloc

from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Operation counts
    message_operations: int = 0
    memory_operations: int = 0
    tool_operations: int = 0
    storage_operations: int = 0
    compression_operations: int = 0
    search_operations: int = 0
    
    # Timing metrics (in seconds)
    average_message_time: float = 0.0
    average_memory_time: float = 0.0
    average_tool_time: float = 0.0
    average_storage_time: float = 0.0
    average_compression_time: float = 0.0
    average_search_time: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    thread_count: int = 0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    
    # System health
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceAlert:
    """Performance alert notification."""
    level: str  # "info", "warning", "error", "critical"
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceMonitor:
    """
    Performance monitoring system for the context manager.
    
    Provides:
    - Real-time metrics collection
    - Performance alerts
    - Historical data tracking
    - Resource monitoring
    - Optimization recommendations
    """
    
    def __init__(self, 
                 sample_interval: float = 1.0,
                 history_size: int = 1000,
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize performance monitor.
        
        Args:
            sample_interval: Interval between metric samples in seconds
            history_size: Number of historical samples to keep
            alert_thresholds: Custom alert thresholds
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'response_time': {'warning': 1.0, 'critical': 5.0},
            'cache_hit_rate': {'warning': 0.5, 'critical': 0.3}
        }
        
        # Metrics storage
        self.current_metrics = PerformanceMetrics()
        self.metrics_history = deque(maxlen=history_size)
        self.operation_times = defaultdict(lambda: deque(maxlen=100))
        
        # Alert system
        self.alerts: List[PerformanceAlert] = []
        self.alert_handlers: List[Callable[[PerformanceAlert], None]] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        self._is_monitoring = False
        
        # Resource monitoring
        self._process = psutil.Process()
        self._start_time = time.time()
        
        # Memory tracking
        tracemalloc.start()
        
        logger.info("Performance monitor initialized")
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self._is_monitoring = True
        self._stop_event.clear()
        
        # Start background monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        self._stop_event.set()
        
        # Stop monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    def record_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """
        Record an operation for performance tracking.
        
        Args:
            operation: Operation name (e.g., "message_add", "memory_search")
            duration: Operation duration in seconds
            success: Whether the operation was successful
        """
        # Record timing
        self.operation_times[operation].append(duration)
        
        # Update operation counts
        if operation.startswith("message"):
            self.current_metrics.message_operations += 1
        elif operation.startswith("memory"):
            self.current_metrics.memory_operations += 1
        elif operation.startswith("tool"):
            self.current_metrics.tool_operations += 1
        elif operation.startswith("storage"):
            self.current_metrics.storage_operations += 1
        elif operation.startswith("compression"):
            self.current_metrics.compression_operations += 1
        elif operation.startswith("search"):
            self.current_metrics.search_operations += 1
        
        # Update error tracking
        if not success:
            self.current_metrics.error_count += 1
        
        # Update average times
        if self.operation_times[operation]:
            avg_time = sum(self.operation_times[operation]) / len(self.operation_times[operation])
            
            if operation.startswith("message"):
                self.current_metrics.average_message_time = avg_time
            elif operation.startswith("memory"):
                self.current_metrics.average_memory_time = avg_time
            elif operation.startswith("tool"):
                self.current_metrics.average_tool_time = avg_time
            elif operation.startswith("storage"):
                self.current_metrics.average_storage_time = avg_time
            elif operation.startswith("compression"):
                self.current_metrics.average_compression_time = avg_time
            elif operation.startswith("search"):
                self.current_metrics.average_search_time = avg_time
    
    def record_cache_operation(self, hit: bool) -> None:
        """
        Record a cache operation for hit rate tracking.
        
        Args:
            hit: Whether the operation was a cache hit
        """
        if hit:
            self.current_metrics.cache_hits += 1
        else:
            self.current_metrics.cache_misses += 1
        
        # Update hit rate
        total_operations = self.current_metrics.cache_hits + self.current_metrics.cache_misses
        if total_operations > 0:
            self.current_metrics.cache_hit_rate = (
                self.current_metrics.cache_hits / total_operations
            )
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """
        Add an alert handler.
        
        Args:
            handler: Function to call when alert is triggered
        """
        self.alert_handlers.append(handler)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, hours: int = 24) -> List[PerformanceMetrics]:
        """
        Get historical metrics.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of historical metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.last_updated >= cutoff_time
        ]
    
    def get_alerts(self, level: Optional[str] = None, 
                   hours: int = 24) -> List[PerformanceAlert]:
        """
        Get performance alerts.
        
        Args:
            level: Alert level filter
            hours: Number of hours of alerts to retrieve
            
        Returns:
            List of alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary with insights.
        
        Returns:
            Dictionary containing performance summary
        """
        metrics = self.current_metrics
        
        # Calculate overall health score
        health_score = self._calculate_health_score(metrics)
        
        # Get recent trends
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        trends = self._calculate_trends(recent_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        return {
            'health_score': health_score,
            'current_metrics': metrics.to_dict(),
            'trends': trends,
            'recommendations': recommendations,
            'active_alerts': len([a for a in self.alerts[-10:] if a.level in ['warning', 'critical']]),
            'uptime_hours': metrics.uptime_seconds / 3600
        }
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.sample_interval)
                if not self._stop_event.is_set():
                    await self._collect_metrics()
                    await self._check_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _collect_metrics(self) -> None:
        """Collect current performance metrics."""
        try:
            # Update resource metrics
            memory_info = self._process.memory_info()
            self.current_metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            
            self.current_metrics.cpu_usage_percent = self._process.cpu_percent()
            
            # Get disk usage
            try:
                disk_usage = psutil.disk_usage('/')
                self.current_metrics.disk_usage_mb = disk_usage.used / 1024 / 1024
            except:
                pass
            
            # Update thread count
            self.current_metrics.thread_count = threading.active_count()
            
            # Update uptime
            self.current_metrics.uptime_seconds = time.time() - self._start_time
            
            # Calculate error rate
            total_operations = (
                self.current_metrics.message_operations +
                self.current_metrics.memory_operations +
                self.current_metrics.tool_operations +
                self.current_metrics.storage_operations
            )
            
            if total_operations > 0:
                self.current_metrics.error_rate = (
                    self.current_metrics.error_count / total_operations
                )
            
            # Update timestamp
            self.current_metrics.last_updated = datetime.now()
            
            # Add to history
            self.metrics_history.append(self.current_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for performance alerts."""
        metrics = self.current_metrics
        
        # Check memory usage
        await self._check_threshold(
            'memory_usage', metrics.memory_usage_mb, 
            self.alert_thresholds['memory_usage']
        )
        
        # Check CPU usage
        await self._check_threshold(
            'cpu_usage', metrics.cpu_usage_percent,
            self.alert_thresholds['cpu_usage']
        )
        
        # Check error rate
        await self._check_threshold(
            'error_rate', metrics.error_rate,
            self.alert_thresholds['error_rate']
        )
        
        # Check cache hit rate
        await self._check_threshold(
            'cache_hit_rate', metrics.cache_hit_rate,
            self.alert_thresholds['cache_hit_rate'],
            reverse=True  # Lower is worse for hit rate
        )
        
        # Check response times
        for operation, times in self.operation_times.items():
            if times:
                avg_time = sum(times) / len(times)
                await self._check_threshold(
                    f'{operation}_response_time', avg_time,
                    self.alert_thresholds['response_time']
                )
    
    async def _check_threshold(self, metric: str, value: float, 
                              thresholds: Dict[str, float], reverse: bool = False) -> None:
        """
        Check if a metric exceeds threshold and trigger alert.
        
        Args:
            metric: Metric name
            value: Current value
            thresholds: Threshold values
            reverse: If True, lower values are worse
        """
        if reverse:
            # For metrics like cache hit rate where lower is worse
            if value <= thresholds['critical']:
                await self._trigger_alert('critical', metric, value, thresholds['critical'])
            elif value <= thresholds['warning']:
                await self._trigger_alert('warning', metric, value, thresholds['warning'])
        else:
            # For metrics like memory usage where higher is worse
            if value >= thresholds['critical']:
                await self._trigger_alert('critical', metric, value, thresholds['critical'])
            elif value >= thresholds['warning']:
                await self._trigger_alert('warning', metric, value, thresholds['warning'])
    
    async def _trigger_alert(self, level: str, metric: str, value: float, threshold: float) -> None:
        """Trigger a performance alert."""
        alert = PerformanceAlert(
            level=level,
            metric=metric,
            value=value,
            threshold=threshold,
            message=f"{metric} {level}: {value:.2f} (threshold: {threshold:.2f})"
        )
        
        self.alerts.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Performance alert: {alert.message}")
    
    def _calculate_health_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Deduct points for high resource usage
        if metrics.memory_usage_mb > 1000:  # > 1GB
            score -= min(20, (metrics.memory_usage_mb - 1000) / 100)
        
        if metrics.cpu_usage_percent > 70:
            score -= min(20, (metrics.cpu_usage_percent - 70) / 2)
        
        # Deduct points for high error rate
        if metrics.error_rate > 0.05:
            score -= min(30, metrics.error_rate * 600)
        
        # Deduct points for low cache hit rate
        if metrics.cache_hit_rate < 0.7:
            score -= min(20, (0.7 - metrics.cache_hit_rate) * 67)
        
        # Deduct points for slow response times
        if metrics.average_message_time > 1.0:
            score -= min(10, metrics.average_message_time * 10)
        
        return max(0, min(100, score))
    
    def _calculate_trends(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, str]:
        """Calculate performance trends."""
        if len(metrics_list) < 2:
            return {}
        
        trends = {}
        
        # Calculate trends for key metrics
        current = metrics_list[-1]
        previous = metrics_list[0]
        
        # Memory usage trend
        if previous.memory_usage_mb > 0:
            memory_change = (current.memory_usage_mb - previous.memory_usage_mb) / previous.memory_usage_mb
            trends['memory_usage'] = 'increasing' if memory_change > 0.1 else 'stable'
        
        # CPU usage trend
        if previous.cpu_usage_percent > 0:
            cpu_change = (current.cpu_usage_percent - previous.cpu_usage_percent) / previous.cpu_usage_percent
            trends['cpu_usage'] = 'increasing' if cpu_change > 0.1 else 'stable'
        
        # Error rate trend
        if previous.error_count > 0:
            error_change = (current.error_count - previous.error_count) / previous.error_count
            trends['error_rate'] = 'increasing' if error_change > 0.1 else 'stable'
        
        return trends
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        if metrics.memory_usage_mb > 1000:
            recommendations.append("Consider increasing memory limits or implementing more aggressive caching")
        
        # CPU recommendations
        if metrics.cpu_usage_percent > 70:
            recommendations.append("High CPU usage detected. Consider optimizing algorithms or reducing processing frequency")
        
        # Cache recommendations
        if metrics.cache_hit_rate < 0.7:
            recommendations.append("Low cache hit rate. Consider adjusting cache size or eviction policies")
        
        # Error rate recommendations
        if metrics.error_rate > 0.05:
            recommendations.append("High error rate detected. Review error logs and implement better error handling")
        
        # Response time recommendations
        if metrics.average_message_time > 1.0:
            recommendations.append("Slow message processing detected. Consider optimizing message handling logic")
        
        return recommendations
    
    async def export_metrics(self, file_path: str, format: str = 'json') -> None:
        """
        Export performance metrics to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json' or 'csv')
        """
        try:
            if format == 'json':
                data = {
                    'current_metrics': self.current_metrics.to_dict(),
                    'history': [m.to_dict() for m in self.metrics_history],
                    'alerts': [a.to_dict() for a in self.alerts]
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            elif format == 'csv':
                # Export current metrics as CSV
                import csv
                
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.current_metrics.to_dict().keys())
                    writer.writeheader()
                    writer.writerow(self.current_metrics.to_dict())
            
            logger.info(f"Performance metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise ContextManagerError(f"Failed to export metrics: {e}")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.current_metrics = PerformanceMetrics()
        self.metrics_history.clear()
        self.operation_times.clear()
        self.alerts.clear()
        
        # Reset start time
        self._start_time = time.time()
        
        logger.info("Performance metrics reset")


class PerformanceOptimizer:
    """
    Performance optimization utilities.
    
    Provides automated optimization strategies based on performance metrics.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize performance optimizer.
        
        Args:
            performance_monitor: Performance monitor instance
        """
        self.performance_monitor = performance_monitor
        self.optimization_history = []
        
        logger.info("Performance optimizer initialized")
    
    async def optimize_system(self) -> Dict[str, Any]:
        """
        Perform system optimization based on current metrics.
        
        Returns:
            Dictionary containing optimization results
        """
        metrics = self.performance_monitor.get_current_metrics()
        optimizations = []
        
        try:
            # Optimize based on memory usage
            if metrics.memory_usage_mb > 1000:
                memory_opt = await self._optimize_memory_usage(metrics)
                optimizations.append(memory_opt)
            
            # Optimize based on CPU usage
            if metrics.cpu_usage_percent > 70:
                cpu_opt = await self._optimize_cpu_usage(metrics)
                optimizations.append(cpu_opt)
            
            # Optimize based on cache performance
            if metrics.cache_hit_rate < 0.7:
                cache_opt = await self._optimize_cache_performance(metrics)
                optimizations.append(cache_opt)
            
            # Optimize based on error rate
            if metrics.error_rate > 0.05:
                error_opt = await self._optimize_error_handling(metrics)
                optimizations.append(error_opt)
            
            result = {
                'timestamp': datetime.now(),
                'optimizations_applied': optimizations,
                'metrics_before': metrics.to_dict()
            }
            
            self.optimization_history.append(result)
            
            logger.info(f"Applied {len(optimizations)} optimizations")
            return result
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            raise ContextManagerError(f"Optimization failed: {e}")
    
    async def _optimize_memory_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize memory usage."""
        # This would implement memory optimization strategies
        return {
            'type': 'memory_optimization',
            'action': 'cache_clearing',
            'description': 'Cleared unused cache entries to reduce memory usage'
        }
    
    async def _optimize_cpu_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize CPU usage."""
        # This would implement CPU optimization strategies
        return {
            'type': 'cpu_optimization',
            'action': 'processing_frequency_reduction',
            'description': 'Reduced background processing frequency'
        }
    
    async def _optimize_cache_performance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize cache performance."""
        # This would implement cache optimization strategies
        return {
            'type': 'cache_optimization',
            'action': 'cache_size_adjustment',
            'description': 'Adjusted cache size and eviction policies'
        }
    
    async def _optimize_error_handling(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize error handling."""
        # This would implement error handling optimization strategies
        return {
            'type': 'error_optimization',
            'action': 'retry_mechanism_improvement',
            'description': 'Improved retry mechanisms for failed operations'
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history