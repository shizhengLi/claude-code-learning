"""
Tests for storage layer implementations.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from context_manager.storage.storage_base import (
    StorageLayer, MemoryLayer, StorageStats
)
from context_manager.storage.cache_layer import (
    CacheLayer, CacheConfig, EvictionPolicy
)
from context_manager.storage.disk_layer import (
    DiskLayer, DiskStorageConfig
)
from context_manager.storage.hierarchical_manager import (
    StorageManager, StorageHierarchyConfig, StorageTier
)


class TestStorageStats:
    """Test cases for StorageStats."""
    
    def test_storage_stats_creation(self):
        """Test storage stats creation."""
        stats = StorageStats()
        
        assert stats.total_items == 0
        assert stats.total_size == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.eviction_count == 0
        assert stats.last_access_time == 0.0
        assert stats.last_write_time == 0.0
    
    def test_hit_rate_no_requests(self):
        """Test hit rate calculation with no requests."""
        stats = StorageStats()
        assert stats.hit_rate() == 0.0
    
    def test_hit_rate_with_requests(self):
        """Test hit rate calculation with requests."""
        stats = StorageStats(hit_count=7, miss_count=3)
        assert stats.hit_rate() == 0.7
    
    def test_storage_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = StorageStats(hit_count=5, miss_count=5)
        stats_dict = stats.to_dict()
        
        assert isinstance(stats_dict, dict)
        assert stats_dict['hit_count'] == 5
        assert stats_dict['miss_count'] == 5
        assert stats_dict['total_items'] == 0


class TestMemoryLayer:
    """Test cases for MemoryLayer."""
    
    @pytest.fixture
    def memory_layer(self):
        """Create memory layer for testing."""
        layer = MemoryLayer(name="test_memory", max_size=1000, max_items=10)
        return layer
    
    @pytest.mark.asyncio
    async def test_initialize(self, memory_layer):
        """Test memory layer initialization."""
        await memory_layer.initialize()
        
        assert memory_layer._is_initialized
        assert len(memory_layer._storage) == 0
        assert len(memory_layer._access_times) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, memory_layer):
        """Test memory layer cleanup."""
        await memory_layer.initialize()
        await memory_layer.set("test_key", "test_value")
        
        await memory_layer.cleanup()
        
        assert not memory_layer._is_initialized
        assert len(memory_layer._storage) == 0
        assert len(memory_layer._access_times) == 0
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, memory_layer):
        """Test basic set and get operations."""
        await memory_layer.initialize()
        
        success = await memory_layer.set("key1", "value1")
        assert success is True
        
        value = await memory_layer.get("key1")
        assert value == "value1"
        
        # Check stats
        assert memory_layer.stats.total_items == 1
        assert memory_layer.stats.hit_count == 1
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory_layer):
        """Test getting nonexistent key."""
        await memory_layer.initialize()
        
        value = await memory_layer.get("nonexistent")
        assert value is None
        
        # Check stats
        assert memory_layer.stats.miss_count == 1
    
    @pytest.mark.asyncio
    async def test_set_with_ttl(self, memory_layer):
        """Test setting value with TTL."""
        await memory_layer.initialize()
        
        await memory_layer.set("key1", "value1", ttl=0.1)
        
        # Should exist initially
        value = await memory_layer.get("key1")
        assert value == "value1"
        
        # Wait for TTL to expire
        await asyncio.sleep(0.15)
        
        # Should be expired now
        value = await memory_layer.get("key1")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_delete(self, memory_layer):
        """Test delete operation."""
        await memory_layer.initialize()
        
        await memory_layer.set("key1", "value1")
        assert await memory_layer.exists("key1") is True
        
        success = await memory_layer.delete("key1")
        assert success is True
        
        assert await memory_layer.exists("key1") is False
        value = await memory_layer.get("key1")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_exists(self, memory_layer):
        """Test exists operation."""
        await memory_layer.initialize()
        
        assert await memory_layer.exists("key1") is False
        
        await memory_layer.set("key1", "value1")
        assert await memory_layer.exists("key1") is True
    
    @pytest.mark.asyncio
    async def test_list_keys(self, memory_layer):
        """Test listing keys."""
        await memory_layer.initialize()
        
        await memory_layer.set("key1", "value1")
        await memory_layer.set("key2", "value2")
        await memory_layer.set("key3", "value3")
        
        keys = await memory_layer.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}
        
        # Test with pattern
        keys = await memory_layer.list_keys("key*")
        assert set(keys) == {"key1", "key2", "key3"}
        
        keys = await memory_layer.list_keys("key1")
        assert keys == ["key1"]
    
    @pytest.mark.asyncio
    async def test_clear(self, memory_layer):
        """Test clear operation."""
        await memory_layer.initialize()
        
        await memory_layer.set("key1", "value1")
        await memory_layer.set("key2", "value2")
        
        success = await memory_layer.clear()
        assert success is True
        
        assert len(await memory_layer.list_keys()) == 0
        assert memory_layer.stats.total_items == 0
    
    @pytest.mark.asyncio
    async def test_size_limit_eviction(self, memory_layer):
        """Test size limit eviction."""
        await memory_layer.initialize()
        
        # Set max size to 100 bytes
        memory_layer.max_size = 100
        
        # Add items that exceed the limit
        large_value = "x" * 50  # 50 bytes
        await memory_layer.set("key1", large_value)
        await memory_layer.set("key2", large_value)
        await memory_layer.set("key3", large_value)
        
        # Should have evicted some items
        assert memory_layer.stats.eviction_count > 0
    
    @pytest.mark.asyncio
    async def test_item_limit_eviction(self, memory_layer):
        """Test item limit eviction."""
        await memory_layer.initialize()
        
        # Set max items to 2
        memory_layer.max_items = 2
        
        await memory_layer.set("key1", "value1")
        await memory_layer.set("key2", "value2")
        await memory_layer.set("key3", "value3")
        
        # Should have evicted some items
        assert memory_layer.stats.eviction_count > 0
        assert len(await memory_layer.list_keys()) <= 2
    
    @pytest.mark.asyncio
    async def test_get_size(self, memory_layer):
        """Test getting storage size."""
        await memory_layer.initialize()
        
        assert memory_layer.get_size() == 0
        
        # Add some data
        await memory_layer.set("key1", "value1")
        assert memory_layer.get_size() > 0


class TestCacheLayer:
    """Test cases for CacheLayer."""
    
    @pytest.fixture
    def cache_layer(self):
        """Create cache layer for testing."""
        config = CacheConfig(
            max_size=1000,
            max_items=10,
            eviction_policy=EvictionPolicy.LRU,
            default_ttl=3600
        )
        return CacheLayer(name="test_cache", config=config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, cache_layer):
        """Test cache layer initialization."""
        await cache_layer.initialize()
        
        assert cache_layer._is_initialized
        assert len(cache_layer._storage) == 0
        assert cache_layer._cleanup_task is not None
    
    @pytest.mark.asyncio
    async def test_cleanup(self, cache_layer):
        """Test cache layer cleanup."""
        await cache_layer.initialize()
        await cache_layer.set("test_key", "test_value")
        
        await cache_layer.cleanup()
        
        assert not cache_layer._is_initialized
        assert len(cache_layer._storage) == 0
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache_layer):
        """Test LRU eviction policy."""
        await cache_layer.initialize()
        
        # Set small limits to trigger eviction
        cache_layer.max_size = 100
        cache_layer.max_items = 2
        
        await cache_layer.set("key1", "value1")
        await cache_layer.set("key2", "value2")
        
        # Access key1 to make it most recently used
        await cache_layer.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        await cache_layer.set("key3", "value3")
        
        # key1 should still exist, key2 should be evicted
        assert await cache_layer.get("key1") is not None
        assert await cache_layer.get("key2") is None
        assert await cache_layer.get("key3") is not None
    
    @pytest.mark.asyncio
    async def test_ttl_cleanup(self, cache_layer):
        """Test TTL cleanup."""
        await cache_layer.initialize()
        
        await cache_layer.set("key1", "value1", ttl=0.1)
        
        # Should exist initially
        assert await cache_layer.exists("key1") is True
        
        # Wait for TTL to expire
        await asyncio.sleep(0.15)
        
        # Should be cleaned up
        assert await cache_layer.exists("key1") is False
    
    def test_get_cache_info(self, cache_layer):
        """Test getting cache information."""
        info = cache_layer.get_cache_info()
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'policy' in info
        assert 'max_size' in info
        assert 'current_size' in info
        assert 'hit_rate' in info


class TestDiskLayer:
    """Test cases for DiskLayer."""
    
    @pytest.fixture
    def disk_layer(self):
        """Create disk layer for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DiskStorageConfig(
                base_path=temp_dir,
                max_file_size=1000,
                max_total_size=10000
            )
            return DiskLayer(name="test_disk", config=config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, disk_layer):
        """Test disk layer initialization."""
        await disk_layer.initialize()
        
        assert disk_layer._is_initialized
        assert disk_layer.base_path.exists()
        assert (disk_layer.base_path / "data").exists()
        assert (disk_layer.base_path / "temp").exists()
        assert (disk_layer.base_path / "backup").exists()
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, disk_layer):
        """Test basic set and get operations."""
        await disk_layer.initialize()
        
        success = await disk_layer.set("key1", "value1")
        assert success is True
        
        value = await disk_layer.get("key1")
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_file_rotation(self, disk_layer):
        """Test file rotation when size limit is reached."""
        await disk_layer.initialize()
        
        # Disable compression for this test to make it more predictable
        disk_layer.config.compression_enabled = False
        
        # Set very small file size limit to force rotation
        disk_layer.config.max_file_size = 50
        
        # Add large values to trigger rotation
        large_value = "x" * 100  # 100 bytes
        await disk_layer.set("key1", large_value)
        await disk_layer.set("key2", large_value)
        await disk_layer.set("key3", large_value)
        
        # Should have created multiple files
        data_files = list((disk_layer.base_path / "data").glob("*.dat"))
        assert len(data_files) >= 2
    
    @pytest.mark.asyncio
    async def test_cleanup(self, disk_layer):
        """Test disk layer cleanup."""
        await disk_layer.initialize()
        await disk_layer.set("key1", "value1")
        
        await disk_layer.cleanup()
        
        assert not disk_layer._is_initialized
    
    def test_get_disk_info(self, disk_layer):
        """Test getting disk information."""
        info = disk_layer.get_disk_info()
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'base_path' in info
        assert 'max_file_size' in info
        assert 'current_size' in info


class TestStorageManager:
    """Test cases for StorageManager."""
    
    @pytest.fixture
    def storage_manager(self):
        """Create storage manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageHierarchyConfig(
                disk_path=temp_dir,
                memory_size=1000,
                cache_size=2000,
                promotion_threshold=2
            )
            return StorageManager(config=config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, storage_manager):
        """Test storage manager initialization."""
        await storage_manager.initialize()
        
        assert storage_manager._initialized
        assert storage_manager._memory_layer._is_initialized
        assert storage_manager._cache_layer._is_initialized
        assert storage_manager._disk_layer._is_initialized
    
    @pytest.mark.asyncio
    async def test_get_from_memory(self, storage_manager):
        """Test getting data from memory layer."""
        await storage_manager.initialize()
        
        await storage_manager.set("key1", "value1")
        value = await storage_manager.get("key1")
        
        assert value == "value1"
        # Should be in memory layer
        assert await storage_manager._memory_layer.exists("key1")
    
    @pytest.mark.asyncio
    async def test_get_from_cache(self, storage_manager):
        """Test getting data from cache layer."""
        await storage_manager.initialize()
        
        # Set directly in cache
        await storage_manager._cache_layer.set("key1", "value1")
        
        value = await storage_manager.get("key1")
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_get_from_disk(self, storage_manager):
        """Test getting data from disk layer."""
        await storage_manager.initialize()
        
        # Set directly in disk
        await storage_manager._disk_layer.set("key1", "value1")
        
        value = await storage_manager.get("key1")
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_write_through(self, storage_manager):
        """Test write-through behavior."""
        await storage_manager.initialize()
        
        await storage_manager.set("key1", "value1")
        
        # Should exist in all layers
        assert await storage_manager._memory_layer.exists("key1")
        assert await storage_manager._cache_layer.exists("key1")
        assert await storage_manager._disk_layer.exists("key1")
    
    @pytest.mark.asyncio
    async def test_delete_from_all_layers(self, storage_manager):
        """Test deletion from all layers."""
        await storage_manager.initialize()
        
        await storage_manager.set("key1", "value1")
        success = await storage_manager.delete("key1")
        
        assert success is True
        assert not await storage_manager._memory_layer.exists("key1")
        assert not await storage_manager._cache_layer.exists("key1")
        assert not await storage_manager._disk_layer.exists("key1")
    
    @pytest.mark.asyncio
    async def test_list_keys_all_layers(self, storage_manager):
        """Test listing keys from all layers."""
        await storage_manager.initialize()
        
        await storage_manager._memory_layer.set("mem_key", "value")
        await storage_manager._cache_layer.set("cache_key", "value")
        await storage_manager._disk_layer.set("disk_key", "value")
        
        keys = await storage_manager.list_keys()
        
        assert "mem_key" in keys
        assert "cache_key" in keys
        assert "disk_key" in keys
    
    @pytest.mark.asyncio
    async def test_get_stats(self, storage_manager):
        """Test getting storage statistics."""
        await storage_manager.initialize()
        
        await storage_manager.set("key1", "value1")
        await storage_manager.get("key1")
        
        stats = storage_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total' in stats
        assert 'memory' in stats
        assert 'cache' in stats
        assert 'disk' in stats
    
    @pytest.mark.asyncio
    async def test_cleanup(self, storage_manager):
        """Test storage manager cleanup."""
        await storage_manager.initialize()
        
        await storage_manager.cleanup()
        
        assert not storage_manager._initialized
        assert not storage_manager._memory_layer._is_initialized
        assert not storage_manager._cache_layer._is_initialized
        assert not storage_manager._disk_layer._is_initialized