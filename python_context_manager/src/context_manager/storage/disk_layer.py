"""
Disk layer implementation for persistent storage.

This module provides persistent storage on disk with support for
compression, encryption, and efficient file management.
"""

import asyncio
import json
import pickle
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, asdict
import hashlib
import threading
from contextlib import contextmanager

from .storage_base import StorageLayer, StorageStats
from ..utils.helpers import (
    validate_path, calculate_hash, format_bytes, ContextManagerError, 
    StorageError, CompressionError
)
from ..utils.logging import get_logger
from ..compression import CompressionManager


logger = get_logger(__name__)


@dataclass
class DiskStorageConfig:
    """Disk storage configuration."""
    base_path: Union[str, Path]
    max_file_size: int = 100 * 1024 * 1024  # 100MB per file
    max_total_size: int = 10 * 1024 * 1024 * 1024  # 10GB total
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress files larger than 1KB
    cleanup_interval: float = 3600.0  # 1 hour
    backup_enabled: bool = True
    backup_interval: float = 86400.0  # 24 hours
    file_format: str = "pickle"  # "pickle" or "json"


class DiskLayer(StorageLayer):
    """Disk storage layer implementation."""
    
    def __init__(self, name: str = "disk", config: Optional[DiskStorageConfig] = None):
        """
        Initialize disk layer.
        
        Args:
            name: Name of the disk layer
            config: Disk storage configuration
        """
        super().__init__(name, config.max_total_size if config else None)
        self.config = config or DiskStorageConfig(base_path="/tmp/context_manager_disk")
        self.base_path = Path(self.config.base_path)
        
        # Compression manager
        self.compression_manager = CompressionManager()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
        self._stop_tasks = threading.Event()
        
        # File management
        self._current_file_size = 0
        self._current_file_path: Optional[Path] = None
        self._file_lock = threading.Lock()
        
    async def initialize(self) -> None:
        """Initialize the disk layer."""
        try:
            # Create base directory
            validate_path(self.base_path, must_exist=False, create_dirs=True)
            
            # Create subdirectories
            (self.base_path / "data").mkdir(exist_ok=True)
            (self.base_path / "temp").mkdir(exist_ok=True)
            (self.base_path / "backup").mkdir(exist_ok=True)
            
            # Initialize current file
            await self._initialize_current_file()
            
            # Start background tasks
            self._stop_tasks.clear()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            if self.config.backup_enabled:
                self._backup_task = asyncio.create_task(self._backup_loop())
            
            self._is_initialized = True
            logger.info(f"Disk layer '{self.name}' initialized at {self.base_path}")
            
        except Exception as e:
            raise StorageError(f"Failed to initialize disk layer: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Stop background tasks
        if self._cleanup_task:
            self._stop_tasks.set()
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._backup_task:
            self._stop_tasks.set()
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        
        # Close current file
        with self._file_lock:
            self._current_file_path = None
            self._current_file_size = 0
        
        self._is_initialized = False
        logger.info(f"Disk layer '{self.name}' cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if not self._is_initialized:
            raise StorageError("Disk layer not initialized")
        
        with self._lock_context():
            try:
                # Find file containing the key
                file_path = await self._find_file_for_key(key)
                if not file_path:
                    self._update_stats_on_access(False)
                    return None
                
                # Read file
                data = await self._read_file(file_path)
                
                # Get value
                if key in data:
                    self._update_stats_on_access(True)
                    # Return just the value, not the metadata
                    item = data[key]
                    if isinstance(item, dict) and 'value' in item:
                        return item['value']
                    return item
                
                self._update_stats_on_access(False)
                return None
                
            except Exception as e:
                logger.error(f"Error getting key '{key}' from disk: {e}")
                self._update_stats_on_access(False)
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value by key."""
        if not self._is_initialized:
            raise StorageError("Disk layer not initialized")
        
        with self._lock_context():
            try:
                # Calculate size
                value_size = self._calculate_size(value)
                
                # Read current file data
                data = {}
                if self._current_file_path and self._current_file_path.exists():
                    data = await self._read_file(self._current_file_path)
                
                # Check if we need to rotate files
                # Calculate potential new size (accounting for overwrite)
                old_size = 0
                if key in data:
                    old_size = self._calculate_size(data.get(key, None))
                
                potential_new_size = self._current_file_size - old_size + value_size
                if potential_new_size > self.config.max_file_size:
                    await self._rotate_file()
                    # After rotation, data is empty
                    data = {}
                    old_size = 0
                
                # Check total size limit
                if not self._check_size_limit(value_size):
                    await self._cleanup_old_files(value_size)
                data[key] = {
                    'value': value,
                    'created_at': time.time(),
                    'ttl': ttl,
                    'size': value_size
                }
                
                # Write to file
                await self._write_file(self._current_file_path, data)
                
                # Update size tracking with actual file size
                if self._current_file_path and self._current_file_path.exists():
                    self._current_file_size = self._current_file_path.stat().st_size
                else:
                    self._current_file_size += value_size - old_size
                
                # Update statistics
                size_change = value_size - old_size
                self._update_stats_on_write(size_change)
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting key '{key}' to disk: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        if not self._is_initialized:
            raise StorageError("Disk layer not initialized")
        
        with self._lock_context():
            try:
                # Find file containing the key
                file_path = await self._find_file_for_key(key)
                if not file_path:
                    return False
                
                # Read file
                data = await self._read_file(file_path)
                
                if key in data:
                    old_size = self._calculate_size(data[key]['value'])
                    del data[key]
                    
                    # Write back to file
                    if data:
                        await self._write_file(file_path, data)
                    else:
                        # Remove empty file
                        file_path.unlink()
                    
                    # Update statistics
                    self._update_stats_on_write(-old_size)
                    
                    # Update current file size if needed
                    if file_path == self._current_file_path:
                        self._current_file_size -= old_size
                    
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error deleting key '{key}' from disk: {e}")
                return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._is_initialized:
            raise StorageError("Disk layer not initialized")
        
        with self._lock_context():
            try:
                file_path = await self._find_file_for_key(key)
                if not file_path:
                    return False
                
                data = await self._read_file(file_path)
                return key in data
                
            except Exception as e:
                logger.error(f"Error checking existence of key '{key}': {e}")
                return False
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        if not self._is_initialized:
            raise StorageError("Disk layer not initialized")
        
        with self._lock_context():
            keys = []
            try:
                # Scan all data files
                data_dir = self.base_path / "data"
                for file_path in data_dir.glob("*.dat"):
                    try:
                        data = await self._read_file(file_path)
                        file_keys = [k for k in data.keys() if self._matches_pattern(k, pattern)]
                        keys.extend(file_keys)
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                
                return keys
                
            except Exception as e:
                logger.error(f"Error listing keys: {e}")
                return []
    
    async def clear(self) -> bool:
        """Clear all data from disk storage."""
        if not self._is_initialized:
            raise StorageError("Disk layer not initialized")
        
        with self._lock_context():
            try:
                # Remove all data files
                data_dir = self.base_path / "data"
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                    data_dir.mkdir(exist_ok=True)
                
                # Reset current file
                await self._initialize_current_file()
                
                # Update statistics
                old_size = self.stats.total_size
                self._update_stats_on_write(-old_size)
                
                return True
                
            except Exception as e:
                logger.error(f"Error clearing disk storage: {e}")
                return False
    
    def get_size(self) -> int:
        """Get current storage size in bytes."""
        try:
            total_size = 0
            data_dir = self.base_path / "data"
            if data_dir.exists():
                for file_path in data_dir.glob("*.dat"):
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0
    
    async def _initialize_current_file(self) -> None:
        """Initialize current data file."""
        with self._file_lock:
            import time
            timestamp = int(time.time() * 1000)  # Use milliseconds for uniqueness
            import random
            random_suffix = random.randint(1000, 9999)
            self._current_file_path = self.base_path / "data" / f"data_{timestamp}_{random_suffix}.dat"
            self._current_file_size = 0
    
    async def _rotate_file(self) -> None:
        """Rotate to a new data file."""
        logger.debug(f"Rotating file from {self._current_file_path}")
        await self._initialize_current_file()
    
    async def _find_file_for_key(self, key: str) -> Optional[Path]:
        """Find which file contains the given key."""
        try:
            # Use hash to determine which file might contain the key
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            # Search in files that might contain this key
            data_dir = self.base_path / "data"
            for file_path in sorted(data_dir.glob("*.dat"), reverse=True):
                try:
                    data = await self._read_file(file_path)
                    if key in data:
                        return file_path
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding file for key '{key}': {e}")
            return None
    
    async def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read data from file."""
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Decompress if needed
            if self.config.compression_enabled and len(data) > self.config.compression_threshold:
                try:
                    data = self.compression_manager.decompress(data)
                except CompressionError:
                    pass  # File might not be compressed
            
            # Deserialize
            if self.config.file_format == "pickle":
                return pickle.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}
    
    async def _write_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write data to file."""
        try:
            # Serialize
            if self.config.file_format == "pickle":
                serialized_data = pickle.dumps(data)
            else:
                serialized_data = json.dumps(data, indent=2).encode('utf-8')
            
            # Compress if needed
            if self.config.compression_enabled and len(serialized_data) > self.config.compression_threshold:
                try:
                    serialized_data = self.compression_manager.compress(serialized_data)
                except CompressionError:
                    pass  # Skip compression if it fails
            
            # Write to temporary file first
            temp_path = self.base_path / "temp" / f"temp_{time.time()}.tmp"
            with open(temp_path, 'wb') as f:
                f.write(serialized_data)
            
            # Atomic move
            temp_path.replace(file_path)
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise StorageError(f"Failed to write file: {e}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if value is None:
                return 0
            return len(pickle.dumps(value))
        except (pickle.PicklingError, TypeError):
            return len(str(value).encode('utf-8'))
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        if pattern == "*":
            return True
        
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def _cleanup_old_files(self, required_size: int) -> None:
        """Clean up old files to free up space."""
        try:
            data_dir = self.base_path / "data"
            files = list(data_dir.glob("*.dat"))
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x.stat().st_mtime)
            
            freed_size = 0
            for file_path in files:
                if file_path == self._current_file_path:
                    continue
                
                file_size = file_path.stat().st_size
                file_path.unlink()
                freed_size += file_size
                self.stats.eviction_count += 1
                
                if freed_size >= required_size:
                    break
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_tasks.is_set():
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                if not self._stop_tasks.is_set():
                    await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in disk cleanup loop: {e}")
    
    async def _backup_loop(self) -> None:
        """Background backup loop."""
        while not self._stop_tasks.is_set():
            try:
                await asyncio.sleep(self.config.backup_interval)
                if not self._stop_tasks.is_set():
                    await self._create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in disk backup loop: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired items."""
        current_time = time.time()
        
        try:
            data_dir = self.base_path / "data"
            for file_path in data_dir.glob("*.dat"):
                try:
                    data = await self._read_file(file_path)
                    
                    # Remove expired items
                    expired_keys = [
                        key for key, item in data.items()
                        if item.get('ttl') and current_time > item['ttl']
                    ]
                    
                    if expired_keys:
                        for key in expired_keys:
                            del data[key]
                        
                        if data:
                            await self._write_file(file_path, data)
                        else:
                            file_path.unlink()
                    
                except Exception as e:
                    logger.warning(f"Error cleaning up expired items in {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in cleanup expired: {e}")
    
    async def _create_backup(self) -> None:
        """Create backup of data files."""
        try:
            timestamp = int(time.time())
            backup_dir = self.base_path / "backup" / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy data files
            data_dir = self.base_path / "data"
            for file_path in data_dir.glob("*.dat"):
                shutil.copy2(file_path, backup_dir)
            
            # Clean up old backups (keep last 5)
            backups = sorted((self.base_path / "backup").glob("backup_*"), reverse=True)
            for old_backup in backups[5:]:
                shutil.rmtree(old_backup)
            
            logger.info(f"Created backup at {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def get_disk_info(self) -> Dict[str, Any]:
        """Get detailed disk storage information."""
        try:
            data_dir = self.base_path / "data"
            files = list(data_dir.glob("*.dat"))
            
            return {
                'name': self.name,
                'base_path': str(self.base_path),
                'max_file_size': self.config.max_file_size,
                'max_total_size': self.config.max_total_size,
                'current_size': self.get_size(),
                'file_count': len(files),
                'compression_enabled': self.config.compression_enabled,
                'file_format': self.config.file_format,
                'stats': self.stats.to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return {}