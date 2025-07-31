"""
Python Context Manager

A Python implementation of Claude Code's context management and memory management system.

This package provides:
- Context management with intelligent compression
- Three-layer memory architecture (short-term, medium-term, long-term)
- Tool execution system with parallel processing
- Multi-level storage and caching
- Performance optimization and monitoring
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.context_manager import ContextManager
from .core.memory_manager import MemoryManager
from .core.state_controller import StateController
from .core.config import ConfigManager, ContextManagerConfig

__all__ = [
    "ContextManager",
    "MemoryManager", 
    "StateController",
    "ConfigManager",
    "ContextManagerConfig",
]