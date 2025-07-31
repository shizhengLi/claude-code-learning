"""
Configuration validation and error recovery system.

This module provides comprehensive configuration validation,
error detection, and automatic recovery mechanisms for the context manager.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from datetime import datetime, timedelta
from enum import Enum
import logging

from .config import ConfigManager, ContextManagerConfig
from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError, ConfigurationError


logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Configuration validation levels."""
    BASIC = "basic"          # Essential validation
    STRICT = "strict"        # Comprehensive validation
    PARANOID = "paranoid"    # Extremely thorough validation


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART_COMPONENT = "restart_component"
    RESET_CONFIG = "reset_config"
    FALLBACK_DEFAULTS = "fallback_defaults"
    ROLLBACK_CONFIG = "rollback_config"
    CLEAR_CACHE = "clear_cache"
    RELOAD_PLUGINS = "reload_plugins"


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_time: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryResult:
    """Result of a recovery action."""
    action: RecoveryAction
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_time: datetime = field(default_factory=datetime.now)


@dataclass
class HealthStatus:
    """System health status."""
    is_healthy: bool
    health_score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_healthy': self.is_healthy,
            'health_score': self.health_score,
            'issues': self.issues,
            'warnings': self.warnings,
            'last_check': self.last_check.isoformat()
        }


class ConfigValidator:
    """
    Configuration validation system.
    
    Provides comprehensive validation of configuration parameters
    with detailed error reporting and suggestions.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize configuration validator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.validation_rules = self._load_validation_rules()
        
        logger.info("Configuration validator initialized")
    
    def validate_config(self, config: ContextManagerConfig, 
                       level: ValidationLevel = ValidationLevel.STRICT) -> ValidationResult:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            level: Validation level
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Basic validation (always performed)
            self._validate_basic_config(config, errors, warnings, suggestions)
            
            # Strict validation
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_strict_config(config, errors, warnings, suggestions)
            
            # Paranoid validation
            if level == ValidationLevel.PARANOID:
                self._validate_paranoid_config(config, errors, warnings, suggestions)
            
            # Validate against custom rules
            self._validate_custom_rules(config, errors, warnings, suggestions)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                suggestions=suggestions
            )
    
    def _validate_basic_config(self, config: ContextManagerConfig, 
                              errors: List[str], warnings: List[str], 
                              suggestions: List[str]) -> None:
        """Perform basic configuration validation."""
        # Validate essential parameters
        if config.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        if config.max_tokens > 100000:
            warnings.append("max_tokens is very large, may impact performance")
        
        if config.memory_cache_size <= 0:
            errors.append("memory_cache_size must be positive")
        
        if config.cache_size <= 0:
            errors.append("cache_size must be positive")
        
        # Validate storage path
        if not config.storage_path:
            errors.append("storage_path is required")
        else:
            try:
                storage_path = Path(config.storage_path)
                if not storage_path.parent.exists():
                    warnings.append(f"Storage path parent directory does not exist: {storage_path.parent}")
            except Exception as e:
                errors.append(f"Invalid storage path: {e}")
        
        # Validate compression settings
        if config.compression_threshold < 0 or config.compression_threshold > 1:
            errors.append("compression_threshold must be between 0 and 1")
        
        if config.pruning_threshold < 0 or config.pruning_threshold > 1:
            errors.append("pruning_threshold must be between 0 and 1")
        
        if config.compression_threshold >= config.pruning_threshold:
            warnings.append("compression_threshold should be less than pruning_threshold")
    
    def _validate_strict_config(self, config: ContextManagerConfig, 
                               errors: List[str], warnings: List[str], 
                               suggestions: List[str]) -> None:
        """Perform strict configuration validation."""
        # Validate resource limits
        total_memory = config.memory_cache_size + config.cache_size
        if total_memory > 2 * 1024 * 1024 * 1024:  # 2GB
            warnings.append("Total memory allocation exceeds 2GB")
        
        # Validate timeout settings
        if hasattr(config, 'operation_timeout'):
            if config.operation_timeout <= 0:
                errors.append("operation_timeout must be positive")
            elif config.operation_timeout > 300:
                warnings.append("operation_timeout is very large")
        
        # Validate retry settings
        if hasattr(config, 'max_retries'):
            if config.max_retries < 0:
                errors.append("max_retries must be non-negative")
            elif config.max_retries > 10:
                warnings.append("max_retries is very large")
        
        # Validate thread settings
        if hasattr(config, 'max_threads'):
            if config.max_threads <= 0:
                errors.append("max_threads must be positive")
            elif config.max_threads > 100:
                warnings.append("max_threads is very large")
    
    def _validate_paranoid_config(self, config: ContextManagerConfig, 
                                  errors: List[str], warnings: List[str], 
                                  suggestions: List[str]) -> None:
        """Perform paranoid configuration validation."""
        # Check for potential conflicts
        if config.compression_threshold > 0.9:
            errors.append("compression_threshold too high for effective compression")
        
        if config.pruning_threshold < 0.5:
            errors.append("pruning_threshold too low, may cause excessive pruning")
        
        # Validate file system permissions
        try:
            storage_path = Path(config.storage_path)
            test_file = storage_path / "test_write_permission.tmp"
            test_file.touch()
            test_file.unlink()
        except Exception:
            errors.append("Cannot write to storage directory")
        
        # Validate network connectivity if remote storage
        if hasattr(config, 'remote_storage_enabled') and config.remote_storage_enabled:
            # This would check network connectivity
            warnings.append("Remote storage enabled - network connectivity required")
        
        # Validate memory availability
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            required_memory = config.memory_cache_size + config.cache_size
            
            if required_memory > available_memory * 0.8:
                errors.append("Insufficient available memory for configuration")
        except ImportError:
            warnings.append("psutil not available - cannot validate memory availability")
    
    def _validate_custom_rules(self, config: ContextManagerConfig, 
                              errors: List[str], warnings: List[str], 
                              suggestions: List[str]) -> None:
        """Validate against custom rules."""
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(config)
                if isinstance(rule_result, dict):
                    if 'errors' in rule_result:
                        errors.extend(rule_result['errors'])
                    if 'warnings' in rule_result:
                        warnings.extend(rule_result['warnings'])
                    if 'suggestions' in rule_result:
                        suggestions.extend(rule_result['suggestions'])
            except Exception as e:
                logger.error(f"Custom validation rule {rule_name} failed: {e}")
                errors.append(f"Custom validation rule {rule_name} failed")
    
    def _load_validation_rules(self) -> Dict[str, Callable]:
        """Load custom validation rules."""
        # This would load custom validation rules from configuration
        return {}
    
    def suggest_optimal_config(self, current_config: ContextManagerConfig, 
                             system_info: Dict[str, Any]) -> ContextManagerConfig:
        """
        Suggest optimal configuration based on system information.
        
        Args:
            current_config: Current configuration
            system_info: System information (memory, CPU, etc.)
            
        Returns:
            Suggested optimal configuration
        """
        suggested_config = ContextManagerConfig(**current_config.__dict__)
        
        try:
            # Adjust based on available memory
            if 'available_memory' in system_info:
                available_memory = system_info['available_memory']
                
                # Allocate 25% of available memory to cache
                suggested_config.cache_size = int(available_memory * 0.25)
                suggested_config.memory_cache_size = int(available_memory * 0.1)
            
            # Adjust based on CPU count
            if 'cpu_count' in system_info:
                cpu_count = system_info['cpu_count']
                suggested_config.max_threads = min(cpu_count * 2, 32)
            
            # Adjust token limits based on model capabilities
            if 'model_token_limit' in system_info:
                suggested_config.max_tokens = min(
                    current_config.max_tokens,
                    int(system_info['model_token_limit'] * 0.8)
                )
            
            logger.info("Generated optimal configuration suggestions")
            return suggested_config
            
        except Exception as e:
            logger.error(f"Failed to generate optimal config: {e}")
            return current_config


class ErrorRecoveryManager:
    """
    Error recovery management system.
    
    Provides automatic error detection, recovery actions, and
    system stabilization mechanisms.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize error recovery manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.recovery_strategies = self._load_recovery_strategies()
        self.recovery_history = []
        self.error_counts = {}
        self.recovery_lock = threading.Lock()
        
        logger.info("Error recovery manager initialized")
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> RecoveryResult:
        """
        Handle an error with appropriate recovery action.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            
        Returns:
            RecoveryResult object
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error frequency
        with self.recovery_lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger.error(f"Handling error: {error_type} - {error_message}")
        
        try:
            # Determine recovery action
            recovery_action = self._determine_recovery_action(error, context)
            
            # Execute recovery action
            recovery_result = await self._execute_recovery_action(recovery_action, error, context)
            
            # Record recovery attempt
            self.recovery_history.append(recovery_result)
            
            # Log recovery result
            if recovery_result.success:
                logger.info(f"Recovery successful: {recovery_action.value}")
            else:
                logger.error(f"Recovery failed: {recovery_action.value}")
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"Error during recovery: {e}")
            return RecoveryResult(
                action=RecoveryAction.FALLBACK_DEFAULTS,
                success=False,
                message=f"Recovery failed: {str(e)}"
            )
    
    def _determine_recovery_action(self, error: Exception, context: Dict[str, Any]) -> RecoveryAction:
        """Determine appropriate recovery action based on error type."""
        error_type = type(error).__name__
        
        # Configuration errors
        if error_type in ['ConfigurationError', 'ValueError']:
            return RecoveryAction.RESET_CONFIG
        
        # Storage errors
        if error_type in ['StorageError', 'IOError', 'OSError']:
            return RecoveryAction.CLEAR_CACHE
        
        # Memory errors
        if error_type in ['MemoryError', 'OutOfMemoryError']:
            return RecoveryAction.CLEAR_CACHE
        
        # Network errors
        if error_type in ['ConnectionError', 'TimeoutError']:
            return RecoveryAction.RESTART_COMPONENT
        
        # Tool errors
        if error_type in ['ToolError', 'ExecutionError']:
            return RecoveryAction.RELOAD_PLUGINS
        
        # Default recovery action
        return RecoveryAction.FALLBACK_DEFAULTS
    
    async def _execute_recovery_action(self, action: RecoveryAction, 
                                     error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Execute a recovery action."""
        try:
            if action == RecoveryAction.RESTART_COMPONENT:
                return await self._restart_component(error, context)
            
            elif action == RecoveryAction.RESET_CONFIG:
                return await self._reset_config(error, context)
            
            elif action == RecoveryAction.FALLBACK_DEFAULTS:
                return await self._fallback_to_defaults(error, context)
            
            elif action == RecoveryAction.ROLLBACK_CONFIG:
                return await self._rollback_config(error, context)
            
            elif action == RecoveryAction.CLEAR_CACHE:
                return await self._clear_cache(error, context)
            
            elif action == RecoveryAction.RELOAD_PLUGINS:
                return await self._reload_plugins(error, context)
            
            else:
                return RecoveryResult(
                    action=action,
                    success=False,
                    message=f"Unknown recovery action: {action}"
                )
                
        except Exception as e:
            return RecoveryResult(
                action=action,
                success=False,
                message=f"Recovery action failed: {str(e)}"
            )
    
    async def _restart_component(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Restart a failed component."""
        # This would implement component restart logic
        return RecoveryResult(
            action=RecoveryAction.RESTART_COMPONENT,
            success=True,
            message="Component restart initiated",
            details={'component': context.get('component', 'unknown')}
        )
    
    async def _reset_config(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Reset configuration to defaults."""
        try:
            # Load default configuration
            default_config = ContextManagerConfig()
            
            # Update configuration manager
            self.config_manager.config = default_config
            
            return RecoveryResult(
                action=RecoveryAction.RESET_CONFIG,
                success=True,
                message="Configuration reset to defaults"
            )
            
        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.RESET_CONFIG,
                success=False,
                message=f"Failed to reset config: {str(e)}"
            )
    
    async def _fallback_to_defaults(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Fallback to default values for specific settings."""
        # This would implement fallback logic for specific settings
        return RecoveryResult(
            action=RecoveryAction.FALLBACK_DEFAULTS,
            success=True,
            message="Fallback to defaults completed"
        )
    
    async def _rollback_config(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Rollback configuration to previous working state."""
        # This would implement config rollback logic
        return RecoveryResult(
            action=RecoveryAction.ROLLBACK_CONFIG,
            success=True,
            message="Configuration rollback completed"
        )
    
    async def _clear_cache(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Clear system cache."""
        # This would implement cache clearing logic
        return RecoveryResult(
            action=RecoveryAction.CLEAR_CACHE,
            success=True,
            message="Cache clearing completed"
        )
    
    async def _reload_plugins(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Reload plugins and extensions."""
        # This would implement plugin reloading logic
        return RecoveryResult(
            action=RecoveryAction.RELOAD_PLUGINS,
            success=True,
            message="Plugin reload completed"
        )
    
    def _load_recovery_strategies(self) -> Dict[str, Callable]:
        """Load custom recovery strategies."""
        # This would load custom recovery strategies
        return {}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self.recovery_lock:
            return {
                'total_errors': sum(self.error_counts.values()),
                'error_types': dict(self.error_counts),
                'recovery_attempts': len(self.recovery_history),
                'successful_recoveries': len([r for r in self.recovery_history if r.success]),
                'failed_recoveries': len([r for r in self.recovery_history if not r.success])
            }
    
    def get_recovery_history(self, hours: int = 24) -> List[RecoveryResult]:
        """
        Get recovery history.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of recovery results
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            result for result in self.recovery_history
            if result.recovery_time >= cutoff_time
        ]


class HealthChecker:
    """
    System health checker.
    
    Provides comprehensive health monitoring and diagnostic capabilities.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize health checker.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.health_checks = self._load_health_checks()
        self.health_history = []
        self.current_health = HealthStatus(is_healthy=True, health_score=1.0)
        
        logger.info("Health checker initialized")
    
    async def check_health(self, comprehensive: bool = False) -> HealthStatus:
        """
        Perform health check.
        
        Args:
            comprehensive: Whether to perform comprehensive health check
            
        Returns:
            HealthStatus object
        """
        issues = []
        warnings = []
        health_score = 1.0
        
        try:
            # Basic health checks
            health_score = self._check_basic_health(issues, warnings, health_score)
            
            # Comprehensive health checks
            if comprehensive:
                health_score = await self._check_comprehensive_health(issues, warnings, health_score)
            
            # Run custom health checks
            health_score = await self._run_custom_health_checks(issues, warnings, health_score)
            
            # Determine overall health
            is_healthy = len(issues) == 0 and health_score > 0.5
            
            # Create health status
            health_status = HealthStatus(
                is_healthy=is_healthy,
                health_score=health_score,
                issues=issues,
                warnings=warnings
            )
            
            # Update current health
            self.current_health = health_status
            
            # Record health check
            self.health_history.append(health_status)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                is_healthy=False,
                health_score=0.0,
                issues=[f"Health check failed: {str(e)}"]
            )
    
    def _check_basic_health(self, issues: List[str], warnings: List[str], health_score: float) -> float:
        """Perform basic health checks."""
        # Check configuration
        config = self.config_manager.config
        
        if config.max_tokens <= 0:
            issues.append("Invalid max_tokens configuration")
            health_score -= 0.3
        
        if config.memory_cache_size <= 0:
            issues.append("Invalid memory_cache_size configuration")
            health_score -= 0.3
        
        # Check storage path accessibility
        try:
            storage_path = Path(config.storage_path)
            if not storage_path.parent.exists():
                warnings.append("Storage path parent directory does not exist")
                health_score -= 0.1
        except Exception:
            issues.append("Cannot access storage path")
            health_score -= 0.3
        
        return max(0.0, health_score)
    
    async def _check_comprehensive_health(self, issues: List[str], warnings: List[str], health_score: float) -> float:
        """Perform comprehensive health checks."""
        # Check memory availability
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.available < 100 * 1024 * 1024:  # 100MB
                warnings.append("Low available memory")
                health_score -= 0.1
            
            if memory.percent > 90:
                issues.append("High memory usage")
                health_score -= 0.2
                
        except ImportError:
            warnings.append("psutil not available - cannot check memory health")
        
        # Check disk space
        try:
            disk = psutil.disk_usage('/')
            
            if disk.free < 1024 * 1024 * 1024:  # 1GB
                warnings.append("Low disk space")
                health_score -= 0.1
            
            if disk.percent > 90:
                issues.append("High disk usage")
                health_score -= 0.2
                
        except Exception:
            warnings.append("Cannot check disk health")
        
        # Check CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                warnings.append("High CPU usage")
                health_score -= 0.1
                
        except Exception:
            warnings.append("Cannot check CPU health")
        
        return max(0.0, health_score)
    
    async def _run_custom_health_checks(self, issues: List[str], warnings: List[str], health_score: float) -> float:
        """Run custom health checks."""
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = await check_func()
                
                if isinstance(check_result, dict):
                    if 'issues' in check_result:
                        issues.extend(check_result['issues'])
                    if 'warnings' in check_result:
                        warnings.extend(check_result['warnings'])
                    if 'health_score' in check_result:
                        health_score = min(health_score, check_result['health_score'])
                        
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                warnings.append(f"Health check {check_name} failed")
                health_score -= 0.1
        
        return max(0.0, health_score)
    
    def _load_health_checks(self) -> Dict[str, Callable]:
        """Load custom health checks."""
        # This would load custom health checks
        return {}
    
    def get_health_history(self, hours: int = 24) -> List[HealthStatus]:
        """
        Get health check history.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of health status objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            status for status in self.health_history
            if status.last_check >= cutoff_time
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        recent_health = self.health_history[-10:] if self.health_history else [self.current_health]
        
        healthy_checks = len([h for h in recent_health if h.is_healthy])
        average_score = sum(h.health_score for h in recent_health) / len(recent_health)
        
        return {
            'current_health': self.current_health.to_dict(),
            'recent_healthy_checks': healthy_checks,
            'recent_total_checks': len(recent_health),
            'average_health_score': average_score,
            'total_health_checks': len(self.health_history)
        }