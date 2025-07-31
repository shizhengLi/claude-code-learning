"""
Compression manager for storage optimization.

This module provides a unified interface for compression utilities
used throughout the storage system.
"""

import zlib
import gzip
import pickle
import json
from typing import Any, Optional, Union
from enum import Enum

from .context_compressor import ContextCompressor
from ..utils.helpers import CompressionError
from ..utils.logging import get_logger


logger = get_logger(__name__)


class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    PICKLE = "pickle"


class CompressionManager:
    """Manages compression operations for storage."""
    
    def __init__(self, default_type: CompressionType = CompressionType.ZLIB):
        """
        Initialize compression manager.
        
        Args:
            default_type: Default compression type
        """
        self.default_type = default_type
        self.context_compressor = ContextCompressor()
        
        # Compression thresholds
        self.min_size_for_compression = 1024  # 1KB
        self.compression_ratio_threshold = 0.8  # Compress if saves 20%+
    
    def compress(self, data: Union[bytes, str, Any], 
                 compression_type: Optional[CompressionType] = None) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress
            compression_type: Type of compression to use
            
        Returns:
            Compressed data as bytes
            
        Raises:
            CompressionError: If compression fails
        """
        if compression_type is None:
            compression_type = self.default_type
        
        try:
            # Convert to bytes if needed
            if isinstance(data, str):
                input_data = data.encode('utf-8')
            elif isinstance(data, bytes):
                input_data = data
            else:
                # For complex objects, use pickle
                input_data = pickle.dumps(data)
                compression_type = CompressionType.PICKLE
            
            # Check if compression is worthwhile
            if len(input_data) < self.min_size_for_compression:
                return input_data
            
            # Compress based on type
            if compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(input_data)
            elif compression_type == CompressionType.GZIP:
                compressed = gzip.compress(input_data)
            elif compression_type == CompressionType.PICKLE:
                compressed = zlib.compress(input_data)
            else:  # NONE
                return input_data
            
            # Check if compression actually helped
            if len(compressed) >= len(input_data) * self.compression_ratio_threshold:
                return input_data
            
            # Add compression header
            header = f"{compression_type.value}:{len(input_data)}:".encode('utf-8')
            return header + compressed
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Failed to compress data: {e}")
    
    def decompress(self, data: bytes) -> Union[bytes, str, Any]:
        """
        Decompress data.
        
        Args:
            data: Compressed data
            
        Returns:
            Decompressed data
            
        Raises:
            CompressionError: If decompression fails
        """
        try:
            # Check if data has compression header
            if b':' not in data:
                # Assume uncompressed data
                return data
            
            # Parse header
            header_end = data.find(b':', data.find(b':') + 1)
            if header_end == -1:
                return data
            
            header = data[:header_end].decode('utf-8')
            parts = header.split(':')
            
            if len(parts) != 2:
                return data
            
            compression_type = CompressionType(parts[0])
            original_size = int(parts[1])
            
            compressed_data = data[header_end + 1:]
            
            # Decompress based on type
            if compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(compressed_data)
            elif compression_type == CompressionType.GZIP:
                decompressed = gzip.decompress(compressed_data)
            elif compression_type == CompressionType.PICKLE:
                decompressed = pickle.loads(zlib.decompress(compressed_data))
            else:  # NONE
                decompressed = compressed_data
            
            # Verify decompression
            if len(decompressed) != original_size:
                logger.warning(f"Decompressed size mismatch: expected {original_size}, got {len(decompressed)}")
            
            return decompressed
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise CompressionError(f"Failed to decompress data: {e}")
    
    def compress_context(self, context: dict) -> dict:
        """
        Compress context data using context compressor.
        
        Args:
            context: Context dictionary
            
        Returns:
            Compressed context
        """
        return self.context_compressor.compress_context(context)
    
    def decompress_context(self, compressed_context: dict) -> dict:
        """
        Decompress context data.
        
        Args:
            compressed_context: Compressed context dictionary
            
        Returns:
            Decompressed context
        """
        return self.context_compressor.decompress_context(compressed_context)
    
    def get_compression_stats(self, original_data: Union[bytes, str, Any], 
                              compressed_data: bytes) -> dict:
        """
        Get compression statistics.
        
        Args:
            original_data: Original data
            compressed_data: Compressed data
            
        Returns:
            Compression statistics
        """
        try:
            # Get original size
            if isinstance(original_data, str):
                original_size = len(original_data.encode('utf-8'))
            elif isinstance(original_data, bytes):
                original_size = len(original_data)
            else:
                original_size = len(pickle.dumps(original_data))
            
            compressed_size = len(compressed_data)
            
            return {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / original_size if original_size > 0 else 0,
                'space_saved': original_size - compressed_size,
                'space_saved_percent': ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate compression stats: {e}")
            return {
                'original_size': 0,
                'compressed_size': 0,
                'compression_ratio': 0,
                'space_saved': 0,
                'space_saved_percent': 0
            }
    
    def is_compressed(self, data: bytes) -> bool:
        """
        Check if data is compressed.
        
        Args:
            data: Data to check
            
        Returns:
            True if data is compressed
        """
        try:
            return b':' in data and any(
                data.startswith(f"{ct.value}:".encode('utf-8')) 
                for ct in CompressionType if ct != CompressionType.NONE
            )
        except Exception:
            return False
    
    def get_compression_type(self, data: bytes) -> CompressionType:
        """
        Get compression type of data.
        
        Args:
            data: Data to check
            
        Returns:
            Compression type
        """
        try:
            if b':' not in data:
                return CompressionType.NONE
            
            first_colon = data.find(b':')
            header = data[:first_colon].decode('utf-8')
            
            try:
                return CompressionType(header)
            except ValueError:
                return CompressionType.NONE
                
        except Exception:
            return CompressionType.NONE