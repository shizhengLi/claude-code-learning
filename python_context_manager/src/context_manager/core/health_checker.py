"""
Health check and diagnostic tools.

This module provides comprehensive health monitoring, diagnostic capabilities,
and system analysis tools for the context manager.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from pathlib import Path
import traceback
import sys

from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError, HealthCheckError


logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CheckCategory(Enum):
    """Health check categories."""
    SYSTEM = "system"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    SECURITY = "security"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    category: CheckCategory
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category.value,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    health_score: float  # 0.0 to 1.0
    checks: List[HealthCheckResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'health_score': self.health_score,
            'checks': [check.to_dict() for check in self.checks],
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'last_check': self.last_check.isoformat()
        }


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    timestamp: datetime = field(default_factory=datetime.now)
    system_info: Dict[str, Any] = field(default_factory=dict)
    health_status: SystemHealth = field(default_factory=lambda: SystemHealth(HealthStatus.UNKNOWN, 0.0))
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    configuration_analysis: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if hasattr(self, 'timestamp') else datetime.now().isoformat(),
            'system_info': self.system_info,
            'health_status': self.health_status.to_dict(),
            'performance_metrics': self.performance_metrics,
            'resource_usage': self.resource_usage,
            'configuration_analysis': self.configuration_analysis,
            'error_analysis': self.error_analysis,
            'recommendations': self.recommendations
        }


class HealthChecker:
    """
    Comprehensive health checking and diagnostic system.
    
    Provides:
    - Multi-category health checks
    - Performance monitoring
    - Resource usage analysis
    - Error analysis
    - Automated diagnostics
    - Health reporting
    """
    
    def __init__(self, check_interval: float = 60.0):
        """
        Initialize health checker.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: List[SystemHealth] = []
        self.current_health = SystemHealth(HealthStatus.UNKNOWN, 0.0)
        
        # Background task management
        self._background_task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        self._is_running = False
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Health checker initialized")
    
    async def start(self) -> None:
        """Start health checking."""
        if self._is_running:
            logger.warning("Health checker already running")
            return
        
        self._is_running = True
        self._stop_event.clear()
        
        # Start background health checking
        self._background_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Health checker started")
    
    async def stop(self) -> None:
        """Stop health checking."""
        if not self._is_running:
            return
        
        self._is_running = False
        self._stop_event.set()
        
        # Stop background task
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health checker stopped")
    
    def register_health_check(self, name: str, category: CheckCategory, 
                            check_func: Callable) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Health check name
            category: Check category
            check_func: Function that performs the check
        """
        self.health_checks[name] = {
            'category': category,
            'func': check_func
        }
        logger.info(f"Registered health check: {name} ({category})")
    
    async def perform_health_check(self, categories: Optional[List[CheckCategory]] = None) -> SystemHealth:
        """
        Perform comprehensive health check.
        
        Args:
            categories: List of categories to check (None for all)
            
        Returns:
            SystemHealth object
        """
        start_time = time.time()
        checks = []
        issues = []
        warnings = []
        
        try:
            # Perform health checks
            for name, check_info in self.health_checks.items():
                if categories and check_info['category'] not in categories:
                    continue
                
                try:
                    result = await self._execute_health_check(name, check_info)
                    checks.append(result)
                    
                    # Collect issues and warnings
                    if result.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                        issues.append(f"{name}: {result.message}")
                    elif result.status == HealthStatus.WARNING:
                        warnings.append(f"{name}: {result.message}")
                        
                except Exception as e:
                    logger.error(f"Health check {name} failed: {e}")
                    checks.append(HealthCheckResult(
                        name=name,
                        category=check_info['category'],
                        status=HealthStatus.CRITICAL,
                        message=f"Health check failed: {str(e)}"
                    ))
                    issues.append(f"{name}: Health check execution failed")
            
            # Calculate overall health
            health_status, health_score = self._calculate_overall_health(checks)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(checks, issues, warnings)
            
            # Create system health
            system_health = SystemHealth(
                status=health_status,
                health_score=health_score,
                checks=checks,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                last_check=datetime.now()
            )
            
            # Update current health
            self.current_health = system_health
            
            # Add to history
            self.health_history.append(system_health)
            
            # Log health status
            logger.info(f"Health check completed: {health_status.value} (score: {health_score:.2f})")
            
            return system_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth(
                status=HealthStatus.CRITICAL,
                health_score=0.0,
                issues=[f"Health check execution failed: {str(e)}"],
                last_check=datetime.now()
            )
    
    async def generate_diagnostic_report(self) -> DiagnosticReport:
        """
        Generate comprehensive diagnostic report.
        
        Returns:
            DiagnosticReport object
        """
        try:
            # Perform health check
            health_status = await self.perform_health_check()
            
            # Collect system information
            system_info = await self._collect_system_info()
            
            # Collect performance metrics
            performance_metrics = await self._collect_performance_metrics()
            
            # Collect resource usage
            resource_usage = await self._collect_resource_usage()
            
            # Analyze configuration
            configuration_analysis = await self._analyze_configuration()
            
            # Analyze errors
            error_analysis = await self._analyze_errors()
            
            # Generate recommendations
            recommendations = self._generate_diagnostic_recommendations(
                health_status, system_info, performance_metrics, resource_usage
            )
            
            # Create diagnostic report
            report = DiagnosticReport(
                system_info=system_info,
                health_status=health_status,
                performance_metrics=performance_metrics,
                resource_usage=resource_usage,
                configuration_analysis=configuration_analysis,
                error_analysis=error_analysis,
                recommendations=recommendations
            )
            
            logger.info("Diagnostic report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate diagnostic report: {e}")
            raise HealthCheckError(f"Diagnostic report generation failed: {e}")
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """
        Get health check history.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of system health objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            health for health in self.health_history
            if health.last_check >= cutoff_time
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics."""
        if not self.health_history:
            return {'message': 'No health history available'}
        
        recent_health = self.health_history[-24:] if len(self.health_history) >= 24 else self.health_history
        
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = len([h for h in recent_health if h.status == status])
        
        average_score = sum(h.health_score for h in recent_health) / len(recent_health)
        
        return {
            'current_status': self.current_health.status.value,
            'current_score': self.current_health.health_score,
            'status_distribution': status_counts,
            'average_score': average_score,
            'total_checks': len(self.health_history),
            'recent_issues': len(self.current_health.issues),
            'recent_warnings': len(self.current_health.warnings)
        }
    
    async def export_health_report(self, file_path: str, format: str = 'json') -> None:
        """
        Export health report to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json' or 'html')
        """
        try:
            report = await self.generate_diagnostic_report()
            
            if format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)
            
            elif format == 'html':
                html_content = self._generate_html_report(report)
                with open(file_path, 'w') as f:
                    f.write(html_content)
            
            logger.info(f"Health report exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export health report: {e}")
            raise HealthCheckError(f"Report export failed: {e}")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.check_interval)
                if not self._stop_event.is_set():
                    await self.perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _execute_health_check(self, name: str, check_info: Dict[str, Any]) -> HealthCheckResult:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            result = await check_info['func']()
            
            if isinstance(result, HealthCheckResult):
                result.duration = time.time() - start_time
                return result
            elif isinstance(result, dict):
                return HealthCheckResult(
                    name=name,
                    category=check_info['category'],
                    duration=time.time() - start_time,
                    **result
                )
            else:
                return HealthCheckResult(
                    name=name,
                    category=check_info['category'],
                    status=HealthStatus.HEALTHY,
                    message="Check passed",
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=name,
                category=check_info['category'],
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                duration=time.time() - start_time
            )
    
    def _calculate_overall_health(self, checks: List[HealthCheckResult]) -> Tuple[HealthStatus, float]:
        """Calculate overall health status and score."""
        if not checks:
            return HealthStatus.UNKNOWN, 0.0
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        for check in checks:
            weight = self._get_category_weight(check.category)
            score = self._status_to_score(check.status)
            
            weighted_score += score * weight
            total_weight += weight
        
        health_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine status
        if health_score >= 0.9:
            status = HealthStatus.HEALTHY
        elif health_score >= 0.7:
            status = HealthStatus.WARNING
        elif health_score >= 0.5:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        return status, health_score
    
    def _status_to_score(self, status: HealthStatus) -> float:
        """Convert health status to score."""
        status_scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.WARNING: 0.8,
            HealthStatus.DEGRADED: 0.6,
            HealthStatus.CRITICAL: 0.3,
            HealthStatus.UNKNOWN: 0.0
        }
        return status_scores.get(status, 0.0)
    
    def _get_category_weight(self, category: CheckCategory) -> float:
        """Get category weight for health calculation."""
        category_weights = {
            CheckCategory.SYSTEM: 1.0,
            CheckCategory.MEMORY: 0.8,
            CheckCategory.STORAGE: 0.8,
            CheckCategory.NETWORK: 0.6,
            CheckCategory.PERFORMANCE: 0.7,
            CheckCategory.CONFIGURATION: 0.5,
            CheckCategory.SECURITY: 0.9
        }
        return category_weights.get(category, 0.5)
    
    def _generate_recommendations(self, checks: List[HealthCheckResult], 
                                 issues: List[str], warnings: List[str]) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        # High priority recommendations
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            recommendations.append("Address critical health issues immediately")
        
        # Memory recommendations
        memory_checks = [check for check in checks if check.category == CheckCategory.MEMORY]
        if any(check.status != HealthStatus.HEALTHY for check in memory_checks):
            recommendations.append("Review memory usage and allocation")
        
        # Performance recommendations
        performance_checks = [check for check in checks if check.category == CheckCategory.PERFORMANCE]
        if any(check.status != HealthStatus.HEALTHY for check in performance_checks):
            recommendations.append("Optimize performance bottlenecks")
        
        # Storage recommendations
        storage_checks = [check for check in checks if check.category == CheckCategory.STORAGE]
        if any(check.status != HealthStatus.HEALTHY for check in storage_checks):
            recommendations.append("Check storage capacity and performance")
        
        return recommendations
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'hostname': getattr(__import__('socket'), 'gethostname')(),
            'current_time': datetime.now().isoformat(),
            'uptime': time.time()
        }
        
        # Add system-specific information
        try:
            import platform
            info.update({
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            })
        except Exception:
            pass
        
        # Add memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info.update({
                'total_memory': memory.total,
                'available_memory': memory.available,
                'memory_percent': memory.percent
            })
        except ImportError:
            pass
        
        return info
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics.update({
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_used': disk.used,
                'disk_free': disk.free,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            })
            
        except ImportError:
            metrics['error'] = "psutil not available"
        
        return metrics
    
    async def _collect_resource_usage(self) -> Dict[str, Any]:
        """Collect resource usage information."""
        usage = {}
        
        try:
            import psutil
            process = psutil.Process()
            
            # Process-specific resource usage
            with process.oneshot():
                usage.update({
                    'process_memory_rss': process.memory_info().rss,
                    'process_memory_vms': process.memory_info().vms,
                    'process_cpu_percent': process.cpu_percent(),
                    'process_threads': process.num_threads(),
                    'process_handles': process.num_handles() if hasattr(process, 'num_handles') else 0
                })
            
            # System-wide resource usage
            usage.update({
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_cpu_percent': psutil.cpu_percent(),
                'system_disk_percent': psutil.disk_usage('/').percent
            })
            
        except ImportError:
            usage['error'] = "psutil not available"
        
        return usage
    
    async def _analyze_configuration(self) -> Dict[str, Any]:
        """Analyze configuration."""
        # This would analyze the current configuration
        return {
            'status': 'configuration_analysis_placeholder',
            'message': 'Configuration analysis would be implemented here'
        }
    
    async def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        # This would analyze error logs and patterns
        return {
            'status': 'error_analysis_placeholder',
            'message': 'Error analysis would be implemented here'
        }
    
    def _generate_diagnostic_recommendations(self, health_status: SystemHealth,
                                           system_info: Dict[str, Any],
                                           performance_metrics: Dict[str, Any],
                                           resource_usage: Dict[str, Any]) -> List[str]:
        """Generate diagnostic recommendations."""
        recommendations = []
        
        # Health-based recommendations
        recommendations.extend(health_status.recommendations)
        
        # Performance-based recommendations
        if performance_metrics.get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected - consider load balancing")
        
        if performance_metrics.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Resource-based recommendations
        if resource_usage.get('system_memory_percent', 0) > 80:
            recommendations.append("System memory usage is high - consider freeing memory")
        
        return recommendations
    
    def _generate_html_report(self, report: DiagnosticReport) -> str:
        """Generate HTML report."""
        # Simple HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Context Manager Health Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .health-healthy {{ color: green; }}
                .health-warning {{ color: orange; }}
                .health-degraded {{ color: red; }}
                .health-critical {{ color: darkred; }}
                .metric {{ margin: 5px 0; }}
                .recommendation {{ margin: 5px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007cba; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Context Manager Health Report</h1>
                <p>Generated: {report.timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Health Status</h2>
                <p class="health-{report.health_status.status.value}">Status: {report.health_status.status.value.upper()}</p>
                <p>Health Score: {report.health_status.health_score:.2f}</p>
            </div>
            
            <div class="section">
                <h2>System Information</h2>
                <div class="metric">Platform: {report.system_info.get('platform', 'Unknown')}</div>
                <div class="metric">Python Version: {report.system_info.get('python_version', 'Unknown')}</div>
                <div class="metric">Hostname: {report.system_info.get('hostname', 'Unknown')}</div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._format_metrics_html(report.performance_metrics)}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._format_recommendations_html(report.recommendations)}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML."""
        html = ""
        for key, value in metrics.items():
            if key != 'error':
                html += f'<div class="metric">{key}: {value}</div>'
        return html
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations as HTML."""
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        return html
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # System health check
        self.register_health_check(
            "system_health",
            CheckCategory.SYSTEM,
            self._check_system_health
        )
        
        # Memory health check
        self.register_health_check(
            "memory_health",
            CheckCategory.MEMORY,
            self._check_memory_health
        )
        
        # Storage health check
        self.register_health_check(
            "storage_health",
            CheckCategory.STORAGE,
            self._check_storage_health
        )
        
        # Performance health check
        self.register_health_check(
            "performance_health",
            CheckCategory.PERFORMANCE,
            self._check_performance_health
        )
    
    async def _check_system_health(self) -> HealthCheckResult:
        """Check system health."""
        try:
            # Basic system checks
            import sys
            import os
            
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 7):
                return HealthCheckResult(
                    name="system_health",
                    category=CheckCategory.SYSTEM,
                    status=HealthStatus.WARNING,
                    message=f"Python version {python_version} is below recommended 3.7"
                )
            
            # Check file system permissions
            try:
                test_file = "/tmp/context_manager_test.tmp"
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                return HealthCheckResult(
                    name="system_health",
                    category=CheckCategory.SYSTEM,
                    status=HealthStatus.DEGRADED,
                    message=f"File system permission issue: {str(e)}"
                )
            
            return HealthCheckResult(
                name="system_health",
                category=CheckCategory.SYSTEM,
                status=HealthStatus.HEALTHY,
                message="System health check passed"
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_health",
                category=CheckCategory.SYSTEM,
                status=HealthStatus.CRITICAL,
                message=f"System health check failed: {str(e)}"
            )
    
    async def _check_memory_health(self) -> HealthCheckResult:
        """Check memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return HealthCheckResult(
                    name="memory_health",
                    category=CheckCategory.MEMORY,
                    status=HealthStatus.CRITICAL,
                    message=f"High memory usage: {memory.percent}%",
                    details={'memory_percent': memory.percent, 'available_mb': memory.available / 1024 / 1024}
                )
            elif memory.percent > 80:
                return HealthCheckResult(
                    name="memory_health",
                    category=CheckCategory.MEMORY,
                    status=HealthStatus.WARNING,
                    message=f"Elevated memory usage: {memory.percent}%",
                    details={'memory_percent': memory.percent, 'available_mb': memory.available / 1024 / 1024}
                )
            
            return HealthCheckResult(
                name="memory_health",
                category=CheckCategory.MEMORY,
                status=HealthStatus.HEALTHY,
                message="Memory health check passed",
                details={'memory_percent': memory.percent, 'available_mb': memory.available / 1024 / 1024}
            )
            
        except ImportError:
            return HealthCheckResult(
                name="memory_health",
                category=CheckCategory.MEMORY,
                status=HealthStatus.UNKNOWN,
                message="Cannot check memory health - psutil not available"
            )
    
    async def _check_storage_health(self) -> HealthCheckResult:
        """Check storage health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            if disk.percent > 90:
                return HealthCheckResult(
                    name="storage_health",
                    category=CheckCategory.STORAGE,
                    status=HealthStatus.CRITICAL,
                    message=f"High disk usage: {disk.percent}%",
                    details={'disk_percent': disk.percent, 'free_gb': disk.free / 1024 / 1024 / 1024}
                )
            elif disk.percent > 80:
                return HealthCheckResult(
                    name="storage_health",
                    category=CheckCategory.STORAGE,
                    status=HealthStatus.WARNING,
                    message=f"Elevated disk usage: {disk.percent}%",
                    details={'disk_percent': disk.percent, 'free_gb': disk.free / 1024 / 1024 / 1024}
                )
            
            return HealthCheckResult(
                name="storage_health",
                category=CheckCategory.STORAGE,
                status=HealthStatus.HEALTHY,
                message="Storage health check passed",
                details={'disk_percent': disk.percent, 'free_gb': disk.free / 1024 / 1024 / 1024}
            )
            
        except ImportError:
            return HealthCheckResult(
                name="storage_health",
                category=CheckCategory.STORAGE,
                status=HealthStatus.UNKNOWN,
                message="Cannot check storage health - psutil not available"
            )
    
    async def _check_performance_health(self) -> HealthCheckResult:
        """Check performance health."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                return HealthCheckResult(
                    name="performance_health",
                    category=CheckCategory.PERFORMANCE,
                    status=HealthStatus.CRITICAL,
                    message=f"High CPU usage: {cpu_percent}%",
                    details={'cpu_percent': cpu_percent}
                )
            elif cpu_percent > 80:
                return HealthCheckResult(
                    name="performance_health",
                    category=CheckCategory.PERFORMANCE,
                    status=HealthStatus.WARNING,
                    message=f"Elevated CPU usage: {cpu_percent}%",
                    details={'cpu_percent': cpu_percent}
                )
            
            return HealthCheckResult(
                name="performance_health",
                category=CheckCategory.PERFORMANCE,
                status=HealthStatus.HEALTHY,
                message="Performance health check passed",
                details={'cpu_percent': cpu_percent}
            )
            
        except ImportError:
            return HealthCheckResult(
                name="performance_health",
                category=CheckCategory.PERFORMANCE,
                status=HealthStatus.UNKNOWN,
                message="Cannot check performance health - psutil not available"
            )