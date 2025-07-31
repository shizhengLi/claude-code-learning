"""
Hierarchical storage manager that coordinates multiple storage layers.

This module provides a unified interface for accessing data across different
storage layers with automatic data movement between layers based on access patterns.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

from .storage_base import StorageLayer, StorageStats
from .cache_layer import CacheLayer, CacheConfig, EvictionPolicy
from .disk_layer import DiskLayer, DiskStorageConfig
from ..utils.helpers import ContextManagerError, StorageError
from ..utils.logging import get_logger


logger = get_logger(__name__)


class StorageTier(Enum):
    """Storage tiers in the hierarchy."""
    MEMORY = "memory"
    CACHE = "cache"
    DISK = "disk"


@dataclass
class StorageHierarchyConfig:
    """Configuration for storage hierarchy."""
    # Memory layer config
    memory_size: int = 50 * 1024 * 1024  # 50MB
    memory_items: int = 1000
    
    # Cache layer config
    cache_size: int = 200 * 1024 * 1024  # 200MB
    cache_items: int = 5000
    cache_policy: EvictionPolicy = EvictionPolicy.LRU
    cache_ttl: float = 1800.0  # 30 minutes
    
    # Disk layer config
    disk_path: str = "/tmp/context_manager_storage"
    disk_max_file_size: int = 100 * 1024 * 1024  # 100MB
    disk_max_total_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    
    # Promotion/demotion policies
    promotion_threshold: int = 3  # Accesses to promote to cache
    demotion_threshold: float = 0.1  # Hit rate below which to demote
    write_through: bool = True  # Write to all layers
    read_through: bool = True  # Read from lower layers if not found


class StorageManager:
    """Hierarchical storage manager."""
    
    def __init__(self, config: Optional[StorageHierarchyConfig] = None):
        """
        Initialize storage manager.
        
        Args:
            config: Storage hierarchy configuration
        """
        self.config = config or StorageHierarchyConfig()
        self._initialized = False
        
        # Initialize storage layers
        from .storage_base import MemoryLayer
        
        self._memory_layer = MemoryLayer(
            name="memory",
            max_size=self.config.memory_size,
            max_items=self.config.memory_items
        )
        
        self._cache_layer = CacheLayer(
            name="cache",
            config=CacheConfig(
                max_size=self.config.cache_size,
                max_items=self.config.cache_items,
                eviction_policy=self.config.cache_policy,
                default_ttl=self.config.cache_ttl
            )
        )
        
        self._disk_layer = DiskLayer(
            name="disk",
            config=DiskStorageConfig(
                base_path=self.config.disk_path,
                max_file_size=self.config.disk_max_file_size,
                max_total_size=self.config.disk_max_total_size
            )
        )
        
        # Access tracking for promotion/demotion
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._last_access_times: Dict[str, float] = {}
        self._layer_locations: Dict[str, StorageTier] = {}
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._stop_maintenance = threading.Event()
        
        # Statistics
        self._total_stats = StorageStats()
        
    async def initialize(self) -> None:
        """Initialize all storage layers."""
        try:
            await self._memory_layer.initialize()
            await self._cache_layer.initialize()
            await self._disk_layer.initialize()
            
            # Start background maintenance
            self._stop_maintenance.clear()
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            self._initialized = True
            logger.info("Storage manager initialized successfully")
            
        except Exception as e:
            await self.cleanup()
            raise StorageError(f"Failed to initialize storage manager: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all storage layers."""
        # Stop background tasks
        if self._maintenance_task:
            self._stop_maintenance.set()
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Clean up layers
        await self._memory_layer.cleanup()
        await self._cache_layer.cleanup()
        await self._disk_layer.cleanup()
        
        self._initialized = False
        logger.info("Storage manager cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value by key from the storage hierarchy.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None if not found
        """
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        # Try memory first
        value = await self._memory_layer.get(key)
        if value is not None:
            self._track_access(key, StorageTier.MEMORY)
            return value
        
        # Try cache
        value = await self._cache_layer.get(key)
        if value is not None:
            self._track_access(key, StorageTier.CACHE)
            # Promote to memory if accessed enough
            if self._should_promote(key, StorageTier.MEMORY):
                await self._memory_layer.set(key, value)
            return value
        
        # Try disk
        if self.config.read_through:
            value = await self._disk_layer.get(key)
            if value is not None:
                self._track_access(key, StorageTier.DISK)
                # Promote to cache
                await self._cache_layer.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Set value by key in the storage hierarchy.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        success = True
        
        # Always write to memory
        success = await self._memory_layer.set(key, value, ttl) and success
        
        # Write to cache if write-through enabled
        if self.config.write_through:
            success = await self._cache_layer.set(key, value, ttl) and success
        
        # Write to disk if write-through enabled
        if self.config.write_through:
            success = await self._disk_layer.set(key, value, ttl) and success
        
        if success:
            self._track_access(key, StorageTier.MEMORY)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """
        Delete value by key from all storage layers.
        
        Args:
            key: Storage key
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        # Delete from all layers
        memory_deleted = await self._memory_layer.delete(key)
        cache_deleted = await self._cache_layer.delete(key)
        disk_deleted = await self._disk_layer.delete(key)
        
        # Clean up tracking
        self._access_counts.pop(key, None)
        self._last_access_times.pop(key, None)
        self._layer_locations.pop(key, None)
        
        return memory_deleted or cache_deleted or disk_deleted
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in any storage layer.
        
        Args:
            key: Storage key
            
        Returns:
            True if exists, False otherwise
        """
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        return (await self._memory_layer.exists(key) or
                await self._cache_layer.exists(key) or
                await self._disk_layer.exists(key))
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List keys matching pattern from all storage layers.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys
        """
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        # Get keys from all layers and deduplicate
        memory_keys = set(await self._memory_layer.list_keys(pattern))
        cache_keys = set(await self._cache_layer.list_keys(pattern))
        disk_keys = set(await self._disk_layer.list_keys(pattern))
        
        return list(memory_keys | cache_keys | disk_keys)
    
    async def clear(self) -> bool:
        """
        Clear all data from all storage layers.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        memory_cleared = await self._memory_layer.clear()
        cache_cleared = await self._cache_layer.clear()
        disk_cleared = await self._disk_layer.clear()
        
        # Reset tracking
        self._access_counts.clear()
        self._last_access_times.clear()
        self._layer_locations.clear()
        
        return memory_cleared and cache_cleared and disk_cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        
        Returns:
            Dictionary containing statistics for all layers
        """
        return {
            'total': self._total_stats.to_dict(),
            'memory': self._memory_layer.get_stats().to_dict(),
            'cache': self._cache_layer.get_stats().to_dict(),
            'disk': self._disk_layer.get_stats().to_dict(),
            'access_tracking': {
                'tracked_keys': len(self._access_counts),
                'memory_keys': sum(1 for tier in self._layer_locations.values() if tier == StorageTier.MEMORY),
                'cache_keys': sum(1 for tier in self._layer_locations.values() if tier == StorageTier.CACHE),
                'disk_keys': sum(1 for tier in self._layer_locations.values() if tier == StorageTier.DISK),
            }
        }
    
    async def get_layer_info(self) -> Dict[str, Any]:
        """
        Get detailed information about each storage layer.
        
        Returns:
            Dictionary containing layer information
        """
        return {
            'memory': {
                'size': self._memory_layer.get_size(),
                'max_size': self._memory_layer.max_size,
                'items': len(await self._memory_layer.list_keys()) if self._initialized else 0,
                'max_items': self._memory_layer.max_items,
            },
            'cache': self._cache_layer.get_cache_info() if self._initialized else {},
            'disk': self._disk_layer.get_disk_info() if self._initialized else {}
        }
    
    def _track_access(self, key: str, tier: StorageTier) -> None:
        """Track key access for promotion/demotion decisions."""
        self._access_counts[key] += 1
        self._last_access_times[key] = time.time()
        self._layer_locations[key] = tier
    
    def _should_promote(self, key: str, target_tier: StorageTier) -> bool:
        """Check if key should be promoted to target tier."""
        if target_tier == StorageTier.MEMORY:
            return self._access_counts[key] >= self.config.promotion_threshold
        elif target_tier == StorageTier.CACHE:
            return self._access_counts[key] >= max(1, self.config.promotion_threshold // 2)
        return False
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop for data movement."""
        while not self._stop_maintenance.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute
                if not self._stop_maintenance.is_set():
                    await self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in storage maintenance loop: {e}")
    
    async def _perform_maintenance(self) -> None:
        """Perform maintenance tasks like promotion and demotion."""
        try:
            # Promote frequently accessed items
            await self._promote_items()
            
            # Demote infrequently accessed items
            await self._demote_items()
            
            # Clean up old access tracking
            await self._cleanup_access_tracking()
            
        except Exception as e:
            logger.error(f"Error performing storage maintenance: {e}")
    
    async def _promote_items(self) -> None:
        """Promote frequently accessed items to higher tiers."""
        # Promote from cache to memory
        cache_keys = await self._cache_layer.list_keys()
        for key in cache_keys:
            if self._should_promote(key, StorageTier.MEMORY):
                value = await self._cache_layer.get(key)
                if value is not None:
                    await self._memory_layer.set(key, value)
                    logger.debug(f"Promoted {key} from cache to memory")
        
        # Promote from disk to cache
        if self.config.read_through:
            disk_keys = await self._disk_layer.list_keys()
            for key in disk_keys:
                if self._should_promote(key, StorageTier.CACHE):
                    value = await self._disk_layer.get(key)
                    if value is not None:
                        await self._cache_layer.set(key, value)
                        logger.debug(f"Promoted {key} from disk to cache")
    
    async def _demote_items(self) -> None:
        """Demote infrequently accessed items to lower tiers."""
        current_time = time.time()
        
        # Check memory items for demotion
        memory_keys = await self._memory_layer.list_keys()
        for key in memory_keys:
            # Demote if not accessed recently
            if (key in self._last_access_times and 
                current_time - self._last_access_times[key] > 3600):  # 1 hour
                value = await self._memory_layer.get(key)
                if value is not None:
                    await self._cache_layer.set(key, value)
                    await self._memory_layer.delete(key)
                    logger.debug(f"Demoted {key} from memory to cache")
        
        # Check cache items for demotion
        cache_keys = await self._cache_layer.list_keys()
        for key in cache_keys:
            # Demote if hit rate is low
            if (key in self._access_counts and 
                self._access_counts[key] < self.config.promotion_threshold // 4):
                value = await self._cache_layer.get(key)
                if value is not None:
                    await self._disk_layer.set(key, value)
                    await self._cache_layer.delete(key)
                    logger.debug(f"Demoted {key} from cache to disk")
    
    async def _cleanup_access_tracking(self) -> None:
        """Clean up old access tracking data."""
        current_time = time.time()
        max_age = 24 * 3600  # 24 hours
        
        # Remove old tracking data
        keys_to_remove = []
        for key, last_access in self._last_access_times.items():
            if current_time - last_access > max_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._access_counts.pop(key, None)
            self._last_access_times.pop(key, None)
            self._layer_locations.pop(key, None)
        
        if keys_to_remove:
            logger.debug(f"Cleaned up access tracking for {len(keys_to_remove)} keys")


