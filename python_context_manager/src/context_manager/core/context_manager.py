"""
Main ContextManager class for the context management system.

This module provides the primary interface for managing context, memory,
and tool operations in the system.
"""

from typing import List, Dict, Any, Optional
from .config import ConfigManager, ContextManagerConfig
from .models import ContextWindow, Message, SystemState
from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError


logger = get_logger(__name__)


class ContextManager:
    """
    Main context manager class that coordinates all system components.
    
    This class provides the primary interface for:
    - Context window management
    - Memory operations
    - Tool execution
    - Configuration management
    """
    
    def __init__(self, config: Optional[ContextManagerConfig] = None, 
                 config_path: Optional[str] = None):
        """
        Initialize the context manager.
        
        Args:
            config: Optional configuration object
            config_path: Optional path to configuration file
        """
        # Initialize configuration
        if config is None:
            self.config_manager = ConfigManager(config_path)
        else:
            self.config_manager = ConfigManager()
            self.config_manager.config = config
            
        self.config = self.config_manager.config
        
        # Initialize core components
        self.context_window = ContextWindow(max_tokens=self.config.max_tokens)
        self.system_state = SystemState(context_window=self.context_window)
        
        logger.info(f"ContextManager initialized with max_tokens={self.config.max_tokens}")
    
    def add_message(self, role: str, content: str, **kwargs) -> bool:
        """
        Add a message to the context window.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message parameters
            
        Returns:
            True if message was added successfully
        """
        try:
            message = Message(role=role, content=content, **kwargs)
            return self.context_window.add_message(message)
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise ContextManagerError(f"Failed to add message: {e}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary containing context summary
        """
        return {
            "message_count": len(self.context_window.messages),
            "token_count": self.context_window.current_tokens,
            "max_tokens": self.context_window.max_tokens,
            "utilization": self.context_window.get_utilization(),
            "is_full": self.context_window.is_full()
        }
    
    def clear_context(self) -> None:
        """Clear all messages from the context window."""
        self.context_window.messages.clear()
        self.context_window.current_tokens = 0
        logger.info("Context cleared")
    
    def get_system_state(self) -> SystemState:
        """Get the current system state."""
        return self.system_state
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config_manager.update_config(**kwargs)
        self.config = self.config_manager.config
        
        # Update context window if max_tokens changed
        if 'max_tokens' in kwargs:
            self.context_window.max_tokens = kwargs['max_tokens']
            
        logger.info(f"Configuration updated: {kwargs}")
    
    def __str__(self) -> str:
        """String representation of the context manager."""
        return f"ContextManager(messages={len(self.context_window.messages)}, tokens={self.context_window.current_tokens}/{self.context_window.max_tokens})"