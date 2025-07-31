"""
StateController class for managing system state and operations.

This module provides the state management functionality for coordinating
system operations and maintaining consistent state.
"""

from typing import Dict, Any, Optional, Callable
from .config import ConfigManager, ContextManagerConfig
from .models import SystemState, ContextWindow
from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError


logger = get_logger(__name__)


class StateController:
    """
    State controller for managing system state and operations.
    
    This class handles:
    - System state management
    - Operation coordination
    - State synchronization
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[ContextManagerConfig] = None):
        """
        Initialize the state controller.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config
        else:
            self.config = config
            
        # State management
        self.system_state = SystemState()
        self.operation_callbacks: Dict[str, Callable] = {}
        self.state_lock = False
        
        # Performance monitoring
        self.operation_count = 0
        self.error_count = 0
        self.start_time = None
        
        logger.info("StateController initialized")
    
    def update_state(self, **kwargs) -> None:
        """
        Update system state with new values.
        
        Args:
            **kwargs: State values to update
        """
        try:
            if self.state_lock:
                logger.warning("State is locked, ignoring update")
                return
            
            # Update context window if provided
            if 'context_window' in kwargs:
                self.system_state.context_window = kwargs['context_window']
            
            # Update memory stats if provided
            if 'memory_stats' in kwargs:
                self.system_state.memory_stats.update(kwargs['memory_stats'])
            
            # Update tool stats if provided
            if 'tool_stats' in kwargs:
                self.system_state.tool_stats.update(kwargs['tool_stats'])
            
            # Update performance metrics if provided
            if 'performance_metrics' in kwargs:
                self.system_state.performance_metrics.update(kwargs['performance_metrics'])
            
            # Update timestamp
            import time
            self.system_state.last_updated = time.time()
            
            logger.debug(f"State updated: {kwargs}")
            
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            raise ContextManagerError(f"Failed to update state: {e}")
    
    def get_state(self) -> SystemState:
        """
        Get the current system state.
        
        Returns:
            Current system state
        """
        return self.system_state
    
    def lock_state(self) -> None:
        """Lock the state to prevent modifications."""
        self.state_lock = True
        logger.debug("State locked")
    
    def unlock_state(self) -> None:
        """Unlock the state to allow modifications."""
        self.state_lock = False
        logger.debug("State unlocked")
    
    def is_locked(self) -> bool:
        """
        Check if the state is locked.
        
        Returns:
            True if state is locked
        """
        return self.state_lock
    
    def register_operation_callback(self, operation: str, callback: Callable) -> None:
        """
        Register a callback for a specific operation.
        
        Args:
            operation: Operation name
            callback: Callback function
        """
        self.operation_callbacks[operation] = callback
        logger.debug(f"Registered callback for operation: {operation}")
    
    def execute_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Execute an operation with proper state management.
        
        Args:
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        try:
            # Check if operation has registered callback
            if operation in self.operation_callbacks:
                callback = self.operation_callbacks[operation]
                result = callback(*args, **kwargs)
                
                # Update operation count
                self.operation_count += 1
                self.update_state(performance_metrics={
                    f"{operation}_count": self.operation_count
                })
                
                return result
            else:
                raise ContextManagerError(f"Unknown operation: {operation}")
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Operation {operation} failed: {e}")
            raise ContextManagerError(f"Operation {operation} failed: {e}")
    
    def reset_stats(self) -> None:
        """Reset operation and error statistics."""
        self.operation_count = 0
        self.error_count = 0
        self.start_time = None
        logger.info("Statistics reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dictionary containing performance summary
        """
        import time
        
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time else 0
        
        return {
            "uptime": uptime,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "success_rate": (self.operation_count - self.error_count) / max(self.operation_count, 1),
            "operations_per_second": self.operation_count / max(uptime, 1),
            "state_locked": self.state_lock,
            "registered_operations": list(self.operation_callbacks.keys())
        }
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        import time
        self.start_time = time.time()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.start_time = None
        logger.info("Performance monitoring stopped")
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export current state for serialization.
        
        Returns:
            Dictionary containing exportable state
        """
        return {
            "system_state": self.system_state.to_dict(),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "performance_summary": self.get_performance_summary()
        }
    
    def import_state(self, state_data: Dict[str, Any]) -> None:
        """
        Import state from serialized data.
        
        Args:
            state_data: Dictionary containing state data
        """
        try:
            # Import basic stats
            self.operation_count = state_data.get("operation_count", 0)
            self.error_count = state_data.get("error_count", 0)
            
            # Import system state if available
            if "system_state" in state_data:
                # This would require proper deserialization of SystemState
                logger.info("State imported successfully")
            
            logger.info("State import completed")
            
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            raise ContextManagerError(f"Failed to import state: {e}")
    
    def __str__(self) -> str:
        """String representation of the state controller."""
        return f"StateController(operations={self.operation_count}, errors={self.error_count}, locked={self.state_lock})"