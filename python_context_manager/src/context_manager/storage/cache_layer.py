"""
Cache layer implementation for hierarchical storage system.

This module provides a caching layer that sits between memory and disk storage,
providing fast access to frequently used data with automatic eviction policies.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict

from .storage_base import StorageLayer, StorageStats
from ..utils.helpers import ContextManagerError, StorageError
from ..utils.logging import get_logger


logger = get_logger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Cache configuration."""
    max_size: int = 100 * 1024 * 1024  # 100MB default
    max_items: int = 10000
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    default_ttl: float = 3600.0  # 1 hour default
    cleanup_interval: float = 300.0  # 5 minutes
    compression_threshold: int = 1024  # Compress items larger than 1KB


class CacheLayer(StorageLayer):
    """Cache layer implementation with multiple eviction policies."""
    
    def __init__(self, name: str = "cache", config: Optional[CacheConfig] = None):
        """
        Initialize cache layer.
        
        Args:
            name: Name of the cache layer
            config: Cache configuration
        """
        super().__init__(name, config.max_size if config else None)
        self.config = config or CacheConfig()
        self.max_items = self.config.max_items
        
        # Storage structures
        self._storage: OrderedDict = OrderedDict()  # For LRU/FIFO
        self._frequency: Dict[str, int] = {}  # For LFU
        self._expiry_times: Dict[str, float] = {}
        self._sizes: Dict[str, int] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = threading.Event()
        
        # Compression
        self._compression_enabled = True
        
    async def initialize(self) -> None:
        """Initialize the cache layer."""
        self._storage.clear()
        self._frequency.clear()
        self._expiry_times.clear()
        self._sizes.clear()
        
        # Start background cleanup task
        self._stop_cleanup.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._is_initialized = True
        logger.info(f"Cache layer '{self.name}' initialized with {self.config.eviction_policy.value} policy")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._cleanup_task:
            self._stop_cleanup.set()
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._storage.clear()
        self._frequency.clear()
        self._expiry_times.clear()
        self._sizes.clear()
        
        self._is_initialized = False
        logger.info(f"Cache layer '{self.name}' cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if not self._is_initialized:
            raise StorageError("Cache layer not initialized")
        
        with self._lock_context():
            # Check if key exists and not expired
            if not await self._is_valid_key(key):
                self._update_stats_on_access(False)
                return None
            
            # Update access statistics based on policy
            if self.config.eviction_policy == EvictionPolicy.LRU:
                # Move to end (most recently used)
                self._storage.move_to_end(key)
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                # Increment frequency
                self._frequency[key] = self._frequency.get(key, 0) + 1
            
            self._update_stats_on_access(True)
            return self._storage[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value by key."""
        if not self._is_initialized:
            raise StorageError("Cache layer not initialized")
        
        with self._lock_context():
            # Calculate size
            value_size = self._calculate_size(value)
            
            # Check if we need to evict items
            if not self._check_size_limit(value_size):
                await self._evict_items(value_size)
            
            # Check item limit
            if len(self._storage) >= self.max_items:
                await self._evict_items(value_size)
            
            # Remove existing key if present
            old_size = 0
            if key in self._storage:
                old_size = self._sizes.get(key, 0)
                if self.config.eviction_policy == EvictionPolicy.LRU:
                    self._storage.move_to_end(key)
            
            # Set the value
            self._storage[key] = value
            self._sizes[key] = value_size
            
            # Set TTL
            actual_ttl = ttl or self.config.default_ttl
            if actual_ttl > 0:
                self._expiry_times[key] = time.time() + actual_ttl
            
            # Initialize frequency for LFU
            if self.config.eviction_policy == EvictionPolicy.LFU:
                self._frequency[key] = self._frequency.get(key, 0) + 1
            
            # Update statistics
            size_change = value_size - old_size
            self._update_stats_on_write(size_change)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        if not self._is_initialized:
            raise StorageError("Cache layer not initialized")
        
        with self._lock_context():
            if key in self._storage:
                old_size = self._sizes.get(key, 0)
                del self._storage[key]
                self._sizes.pop(key, None)
                self._expiry_times.pop(key, None)
                self._frequency.pop(key, None)
                
                self._update_stats_on_write(-old_size)
                return True
            
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._is_initialized:
            raise StorageError("Cache layer not initialized")
        
        with self._lock_context():
            return await self._is_valid_key(key)
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        if not self._is_initialized:
            raise StorageError("Cache layer not initialized")
        
        with self._lock_context():
            await self._cleanup_expired()
            
            if pattern == "*":
                return list(self._storage.keys())
            
            # Simple pattern matching
            import fnmatch
            return fnmatch.filter(self._storage.keys(), pattern)
    
    async def clear(self) -> bool:
        """Clear all data from cache."""
        if not self._is_initialized:
            raise StorageError("Cache layer not initialized")
        
        with self._lock_context():
            old_size = self.stats.total_size
            self._storage.clear()
            self._frequency.clear()
            self._expiry_times.clear()
            self._sizes.clear()
            
            self._update_stats_on_write(-old_size)
            return True
    
    def get_size(self) -> int:
        """Get current cache size in bytes."""
        return sum(self._sizes.values())
    
    async def _is_valid_key(self, key: str) -> bool:
        """Check if key is valid (exists and not expired)."""
        if key not in self._storage:
            return False
        
        # Check TTL
        if key in self._expiry_times:
            if time.time() > self._expiry_times[key]:
                await self.delete(key)
                return False
        
        return True
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired items."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self._expiry_times.items()
            if expiry_time <= current_time
        ]
        
        for key in expired_keys:
            await self.delete(key)
    
    async def _evict_items(self, required_size: int) -> None:
        """Evict items based on configured policy."""
        if self.config.eviction_policy == EvictionPolicy.LRU:
            await self._evict_lru(required_size)
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            await self._evict_lfu(required_size)
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            await self._evict_fifo(required_size)
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            await self._evict_ttl(required_size)
    
    async def _evict_lru(self, required_size: int) -> None:
        """Evict least recently used items."""
        freed_size = 0
        while freed_size < required_size and self._storage:
            # Get oldest item (first in OrderedDict)
            key, _ = self._storage.popitem(last=False)
            freed_size += self._sizes.pop(key, 0)
            self._expiry_times.pop(key, None)
            self._frequency.pop(key, None)
            self.stats.eviction_count += 1
    
    async def _evict_lfu(self, required_size: int) -> None:
        """Evict least frequently used items."""
        freed_size = 0
        while freed_size < required_size and self._storage:
            # Find item with lowest frequency
            if not self._frequency:
                break
            
            min_freq = min(self._frequency.values())
            lfu_keys = [k for k, v in self._frequency.items() if v == min_freq]
            
            # If multiple items have same frequency, use LRU among them
            key_to_evict = None
            for key in lfu_keys:
                if key in self._storage:
                    key_to_evict = key
                    break
            
            if key_to_evict is None:
                break
            
            # Remove the item
            self._storage.pop(key_to_evict, None)
            freed_size += self._sizes.pop(key_to_evict, 0)
            self._expiry_times.pop(key_to_evict, None)
            self._frequency.pop(key_to_evict, None)
            self.stats.eviction_count += 1
    
    async def _evict_fifo(self, required_size: int) -> None:
        """Evict first-in-first-out items."""
        freed_size = 0
        while freed_size < required_size and self._storage:
            # Get oldest item (first in OrderedDict)
            key, _ = self._storage.popitem(last=False)
            freed_size += self._sizes.pop(key, 0)
            self._expiry_times.pop(key, None)
            self._frequency.pop(key, None)
            self.stats.eviction_count += 1
    
    async def _evict_ttl(self, required_size: int) -> None:
        """Evict expired items first, then use LRU."""
        # First, clean up expired items
        await self._cleanup_expired()
        
        # If still need space, use LRU
        if self.get_size() + required_size > (self.max_size or float('inf')):
            await self._evict_lru(required_size)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            import pickle
            return len(pickle.dumps(value))
        except (pickle.PicklingError, TypeError):
            return len(str(value).encode('utf-8'))
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_cleanup.is_set():
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                if not self._stop_cleanup.is_set():
                    await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            'name': self.name,
            'policy': self.config.eviction_policy.value,
            'max_size': self.max_size,
            'max_items': self.max_items,
            'current_size': self.get_size(),
            'current_items': len(self._storage),
            'hit_rate': self.stats.hit_rate(),
            'eviction_count': self.stats.eviction_count,
            'stats': self.stats.to_dict()
        }