"""
Base storage layer interface and implementations.

This module provides a hierarchical storage system with multiple layers:
- Memory layer: Fast in-memory storage
- Cache layer: Intermediate caching layer
- Disk layer: Persistent storage on disk
"""

import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator
from pathlib import Path
from threading import Lock, RLock
from dataclasses import dataclass, asdict
import pickle
import hashlib
import shutil
from contextlib import contextmanager

from ..utils.helpers import (
    validate_path, calculate_hash, format_bytes, format_duration,
    ContextManagerError, StorageError, CompressionError
)
from ..utils.logging import get_logger
from ..compression import CompressionManager


logger = get_logger(__name__)


@dataclass
class StorageStats:
    """Storage statistics."""
    total_items: int = 0
    total_size: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    last_access_time: float = 0.0
    last_write_time: float = 0.0
    
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class StorageLayer(ABC):
    """Abstract base class for storage layers."""
    
    def __init__(self, name: str, max_size: Optional[int] = None):
        """
        Initialize storage layer.
        
        Args:
            name: Name of the storage layer
            max_size: Maximum size in bytes (None for unlimited)
        """
        self.name = name
        self.max_size = max_size
        self.stats = StorageStats()
        self._lock = RLock()
        self._is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage layer."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value by key.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None if not found
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Set value by key.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete value by key.
        
        Args:
            key: Storage key
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List keys matching pattern.
        
        Args:
            pattern: Key pattern (supports * wildcard)
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all data from storage.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """
        Get current storage size in bytes.
        
        Returns:
            Storage size in bytes
        """
        pass
    
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset storage statistics."""
        self.stats = StorageStats()
    
    def _update_stats_on_access(self, is_hit: bool) -> None:
        """Update statistics on access."""
        self.stats.last_access_time = time.time()
        if is_hit:
            self.stats.hit_count += 1
        else:
            self.stats.miss_count += 1
    
    def _update_stats_on_write(self, size_change: int) -> None:
        """Update statistics on write."""
        self.stats.last_write_time = time.time()
        self.stats.total_size += size_change
        self.stats.total_items = max(0, self.stats.total_items + (1 if size_change > 0 else -1))
    
    def _check_size_limit(self, required_size: int) -> bool:
        """Check if size limit would be exceeded."""
        if self.max_size is None:
            return True
        return (self.get_size() + required_size) <= self.max_size
    
    @contextmanager
    def _lock_context(self):
        """Context manager for thread-safe operations."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


class MemoryLayer(StorageLayer):
    """In-memory storage layer implementation."""
    
    def __init__(self, name: str = "memory", max_size: Optional[int] = None, 
                 max_items: Optional[int] = None):
        """
        Initialize memory layer.
        
        Args:
            name: Name of the storage layer
            max_size: Maximum size in bytes
            max_items: Maximum number of items
        """
        super().__init__(name, max_size)
        self.max_items = max_items
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        
    async def initialize(self) -> None:
        """Initialize the memory layer."""
        with self._lock_context():
            self._storage.clear()
            self._access_times.clear()
            self._is_initialized = True
            logger.info(f"Memory layer '{self.name}' initialized")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock_context():
            self._storage.clear()
            self._access_times.clear()
            self._is_initialized = False
            logger.info(f"Memory layer '{self.name}' cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if not self._is_initialized:
            raise StorageError("Memory layer not initialized")
        
        with self._lock_context():
            if key in self._storage:
                item = self._storage[key]
                # Check TTL
                if item.get('ttl') and time.time() > item['ttl']:
                    await self.delete(key)
                    self._update_stats_on_access(False)
                    return None
                
                self._access_times[key] = time.time()
                self._update_stats_on_access(True)
                return item['value']
            
            self._update_stats_on_access(False)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value by key."""
        if not self._is_initialized:
            raise StorageError("Memory layer not initialized")
        
        with self._lock_context():
            # Calculate size
            try:
                value_size = len(pickle.dumps(value))
            except (pickle.PicklingError, TypeError):
                value_size = len(str(value).encode('utf-8'))
            
            # Check limits
            if not self._check_size_limit(value_size):
                await self._evict_lru(value_size)
            
            if self.max_items and len(self._storage) >= self.max_items:
                await self._evict_lru(value_size)
            
            # Store value
            expiry_time = time.time() + ttl if ttl else None
            
            old_size = 0
            if key in self._storage:
                try:
                    old_size = len(pickle.dumps(self._storage[key]['value']))
                except (pickle.PicklingError, TypeError):
                    old_size = len(str(self._storage[key]['value']).encode('utf-8'))
            
            self._storage[key] = {
                'value': value,
                'ttl': expiry_time,
                'created_at': time.time(),
                'size': value_size
            }
            self._access_times[key] = time.time()
            
            size_change = value_size - old_size
            self._update_stats_on_write(size_change)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        if not self._is_initialized:
            raise StorageError("Memory layer not initialized")
        
        with self._lock_context():
            if key in self._storage:
                old_size = self._storage[key]['size']
                del self._storage[key]
                del self._access_times[key]
                
                self._update_stats_on_write(-old_size)
                return True
            
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._is_initialized:
            raise StorageError("Memory layer not initialized")
        
        with self._lock_context():
            if key in self._storage:
                # Check TTL
                item = self._storage[key]
                if item.get('ttl') and time.time() > item['ttl']:
                    await self.delete(key)
                    return False
                return True
            return False
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        if not self._is_initialized:
            raise StorageError("Memory layer not initialized")
        
        with self._lock_context():
            # Clean up expired items first
            await self._cleanup_expired()
            
            if pattern == "*":
                return list(self._storage.keys())
            
            # Simple pattern matching
            import fnmatch
            return fnmatch.filter(self._storage.keys(), pattern)
    
    async def clear(self) -> bool:
        """Clear all data from storage."""
        if not self._is_initialized:
            raise StorageError("Memory layer not initialized")
        
        with self._lock_context():
            old_size = self.stats.total_size
            self._storage.clear()
            self._access_times.clear()
            self._update_stats_on_write(-old_size)
            # Reset item count explicitly
            self.stats.total_items = 0
            return True
    
    def get_size(self) -> int:
        """Get current storage size in bytes."""
        return sum(item['size'] for item in self._storage.values())
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired items."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self._storage.items()
            if item.get('ttl') and current_time > item['ttl']
        ]
        
        for key in expired_keys:
            await self.delete(key)
    
    async def _evict_lru(self, required_size: int) -> None:
        """Evict least recently used items."""
        if not self._access_times:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
        
        freed_size = 0
        for key, _ in sorted_keys:
            if key in self._storage:
                freed_size += self._storage[key]['size']
                await self.delete(key)
                self.stats.eviction_count += 1
                
                if freed_size >= required_size:
                    break